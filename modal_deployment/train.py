import json
import os
import time

import modal

GPU = "H100"        # বদলাতে চাইলে "A10G" / "L4" / "L40S"
app = modal.App("bmnet-train")
data_vol = modal.Volume.from_name("bodym")
runs_vol = modal.Volume.from_name("bmnet-runs", create_if_missing=True)
image = (modal.Image.debian_slim()
         .pip_install("torch", "torchvision", "numpy", "pandas", "pillow")
         .add_local_python_source("bodym_dataset", "bmnet"))


@app.function(image=image, gpu=GPU, cpu=8, memory=32768, timeout=86400,
              volumes={"/data": data_vol, "/runs": runs_vol})
def train(run="bmnet_v2", iters=150_000, batch=22, lr=1e-3,
          eval_every=1000, log_every=100, workers=6, pretrained=False,
          bn_momentum=0.01, seed=0):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from bmnet import BMnet
    from bodym_dataset import (TARGETS, BodyM, denorm, load_table, make_stats,
                               split_by_subject, to_input)

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out = f"/runs/{run}"
    os.makedirs(out, exist_ok=True)

    tr_df, va_df = split_by_subject(load_table("/data", "train"), 0.1, seed=0)
    stats = make_stats(tr_df)

    # ---- সবকিছু সেভ: কনফিগ, স্প্লিট, stats ----
    json.dump({"run": run, "iters": iters, "batch": batch, "lr": lr,
               "eval_every": eval_every, "log_every": log_every,
               "pretrained": pretrained, "bn_momentum": bn_momentum,
               "seed": seed, "gpu": GPU,
               "n_train_photos": len(tr_df), "n_val_photos": len(va_df)},
              open(f"{out}/config.json", "w"), indent=2)
    tr_df.to_csv(f"{out}/train_split.csv", index=False)
    va_df.to_csv(f"{out}/val_split.csv", index=False)
    json.dump({k: v.tolist() for k, v in stats.items()}, open(f"{out}/stats.json", "w"))

    tr, va = BodyM(tr_df, "/data", stats), BodyM(va_df, "/data", stats)
    tl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                    pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, batch_size=batch * 2, shuffle=False, num_workers=2)

    model = BMnet(pretrained=pretrained).to(dev).to(memory_format=torch.channels_last)
    for mod in model.modules():                              # BN stats দ্রুত আপডেট হোক
        if isinstance(mod, torch.nn.BatchNorm2d):            # (torchvision ডিফল্ট ০.০০০৩ ছিল সমস্যা)
            mod.momentum = bn_momentum
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(iters * 0.75), int(iters * 0.88)], gamma=0.1)
    step, best = 0, float("inf")

    ck = f"{out}/last.pt"
    if os.path.exists(ck):                                   # resume
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); step = s["step"]
        best = float("inf")                                  # পুরনো (ভুল BN-এর) best বাদ
        print("resumed from step", step)

    def prep(sil, meta):
        x = to_input(sil.to(dev, non_blocking=True), meta.to(dev, non_blocking=True))
        return x.contiguous(memory_format=torch.channels_last)

    @torch.no_grad()
    def evaluate():
        model.eval()
        errs = []
        for sil, meta, y in vl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(prep(sil, meta))
            errs.append((denorm(pred.float(), stats) - denorm(y.to(dev), stats)).abs().cpu())
        model.train()
        e = torch.cat(errs).mean(0)                          # per-measurement MAE (cm)
        return e, e.mean().item()

    def batches():
        while True:
            yield from tl

    it, model_t0, acc, n_acc = batches(), time.time(), 0.0, 0
    tr_err = torch.zeros(len(TARGETS), device=dev)           # eval-window train MAE (cm)
    w_err = torch.zeros(len(TARGETS), device=dev)            # print-window train MAE (cm)
    w_n, w_t0 = 0, time.time()

    model.train()
    while step < iters:
        sil, meta, y = next(it)
        x, y = prep(sil, meta), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        loss = (pred.float() - y).abs().mean()               # L1 on z-scored targets
        with torch.no_grad():
            err = (denorm(pred.float(), stats) - denorm(y, stats)).abs().mean(0)
        tr_err += err
        w_err += err
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        step += 1
        acc += loss.detach(); n_acc += 1

        w_n += 1
        if step % log_every == 0:                            # প্রতি log_every step-এ প্রিন্ট
            print(f"step {step} | train {float(w_err.mean() / w_n):.3f} cm | "
                  f"{w_n / (time.time() - w_t0):.1f} it/s", flush=True)
            w_err.zero_(); w_n = 0; w_t0 = time.time()

        if step % eval_every == 0 or step == iters:
            per, mean_cm = evaluate()
            rate = n_acc / (time.time() - model_t0)
            tr_per = (tr_err / n_acc).tolist()
            rec = {"step": step,
                   "train_l1_z": float(acc / n_acc),
                   "train_mae_cm": float(np.mean(tr_per)),
                   "train_per_cm": dict(zip(TARGETS, [round(v, 3) for v in tr_per])),
                   "val_mae_cm": mean_cm,
                   "per_cm": dict(zip(TARGETS, [round(v, 3) for v in per.tolist()])),
                   "it_per_s": rate, "lr": sched.get_last_lr()[0],
                   "bn_momentum": bn_momentum}
            print(f"==> step {step} | train {rec['train_mae_cm']:.3f} cm | val {mean_cm:.3f} cm | "
                  f"chest {rec['per_cm']['chest']:.2f} waist {rec['per_cm']['waist']:.2f} "
                  f"hip {rec['per_cm']['hip']:.2f} | {rate:.2f} it/s", flush=True)
            with open(f"{out}/log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if mean_cm < best:
                best = mean_cm
                torch.save({"model": model.state_dict(), "stats": stats, "step": step,
                            "val_mae_cm": mean_cm}, f"{out}/best.pt")
            if step % 25000 == 0:                            # মাঝের checkpoint
                torch.save({"model": model.state_dict(), "stats": stats, "step": step},
                           f"{out}/ckpt_{step}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step, "best": best,
                        "stats": stats}, ck)
            runs_vol.commit()
            acc, n_acc, model_t0 = 0.0, 0, time.time()
            tr_err.zero_()


@app.local_entrypoint()
def main(run: str = "bmnet_v2", iters: int = 150000, batch: int = 22,
         eval_every: int = 1000, log_every: int = 100, pretrained: bool = False,
         bn_momentum: float = 0.01):
    train.remote(run=run, iters=iters, batch=batch, eval_every=eval_every,
                 log_every=log_every, pretrained=pretrained, bn_momentum=bn_momentum)