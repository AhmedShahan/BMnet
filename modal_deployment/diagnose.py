import modal

app = modal.App("bmnet-diagnose")
data_vol = modal.Volume.from_name("bodym")
runs_vol = modal.Volume.from_name("bmnet-runs")
image = (modal.Image.debian_slim()
         .pip_install("torch", "torchvision", "numpy", "pandas", "pillow")
         .add_local_python_source("bodym_dataset", "bmnet"))


@app.function(image=image, gpu="L4", cpu=8, memory=24576, timeout=3600,
              volumes={"/data": data_vol, "/runs": runs_vol})
def diagnose(run="bmnet_v2", ckpt="last.pt"):
    import copy
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from bmnet import BMnet
    from bodym_dataset import TARGETS, BodyM, denorm, to_input

    runs_vol.reload()
    dev = "cuda"
    s = torch.load(f"/runs/{run}/{ckpt}", map_location=dev, weights_only=False)
    stats = s["stats"]
    print("checkpoint step:", s["step"])

    tr_df = pd.read_csv(f"/runs/{run}/train_split.csv")
    va_df = pd.read_csv(f"/runs/{run}/val_split.csv")
    tr_eval = BodyM(tr_df.sample(600, random_state=0), "/data", stats)
    tr_cal = BodyM(tr_df.sample(1200, random_state=1), "/data", stats)
    va = BodyM(va_df, "/data", stats)

    def fwd(m, sil, meta):
        x = to_input(sil.to(dev), meta.to(dev)).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(x).float()

    @torch.no_grad()
    def mae(m, ds, batch_stats=False):
        m.train() if batch_stats else m.eval()
        errs = []
        for sil, meta, y in DataLoader(ds, batch_size=44, shuffle=batch_stats, num_workers=2):
            errs.append((denorm(fwd(m, sil, meta), stats) - denorm(y.to(dev), stats)).abs().cpu())
        e = torch.cat(errs).mean(0)
        return e.mean().item(), e[TARGETS.index("chest")].item(), \
            e[TARGETS.index("waist")].item(), e[TARGETS.index("hip")].item()

    def show(name, r):
        print(f"{name:42s} mean {r[0]:6.2f} cm | chest {r[1]:6.2f} waist {r[2]:6.2f} hip {r[3]:6.2f}")

    m = BMnet().to(dev).to(memory_format=torch.channels_last)
    m.load_state_dict(s["model"])

    show("train  (eval mode, as trained)", mae(m, tr_eval))
    show("val    (eval mode, as trained)", mae(m, va))
    show("val    (batch-stats BN)", mae(copy.deepcopy(m), va, batch_stats=True))

    m3 = copy.deepcopy(m)                                   # BN stats বাদ দিয়ে নতুন করে বের করা
    for mod in m3.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None
    m3.train()
    with torch.no_grad():
        for sil, meta, _ in DataLoader(tr_cal, batch_size=44, shuffle=True, num_workers=2):
            fwd(m3, sil, meta)
    show("train  (BN stats recalibrated)", mae(m3, tr_eval))
    show("val    (BN stats recalibrated)", mae(m3, va))


@app.local_entrypoint()
def main(run: str = "bmnet_v2", ckpt: str = "last.pt"):
    diagnose.remote(run=run, ckpt=ckpt)