# eval.py  —  modal run eval.py --run bmnet_v2
import modal
from pathlib import Path

HERE = Path(__file__).parent.resolve()

app = modal.App("bmnet-eval")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "pandas", "pillow")
    .add_local_file(HERE / "bodym_dataset.py", "/root/bodym_dataset.py")
    .add_local_file(HERE / "bmnet.py", "/root/bmnet.py")
)
data_vol = modal.Volume.from_name("bodym")
runs_vol = modal.Volume.from_name("bmnet-runs")


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=7200,
              volumes={"/data": data_vol, "/runs": runs_vol})
def evaluate(run="bmnet_v2", ckpt="best.pt", batch=32, workers=6):
    import os, json
    import numpy as np, pandas as pd, torch
    from torch.utils.data import DataLoader
    from bodym_dataset import BodyM, load_table, to_input, TARGETS
    from bmnet import BMnet

    dev = "cuda"
    out = f"/runs/{run}/eval"
    os.makedirs(out, exist_ok=True)

    # stats (train subject mean/std)
    st = json.load(open(f"/runs/{run}/stats.json"))
    y_mean = torch.tensor(st["y_mean"], dtype=torch.float32, device=dev)
    y_std = torch.tensor(st["y_std"], dtype=torch.float32, device=dev)

    # model
    model = BMnet().to(dev).to(memory_format=torch.channels_last)
    ck = torch.load(f"/runs/{run}/{ckpt}", map_location=dev, weights_only=False)
        
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    summary = {}
    for split in ["testA", "testB"]:
        df = load_table("/data", split).reset_index(drop=True)
        ds = BodyM(df, "/data", st)          # <-- সিগনেচার না মিললে শুধু এই লাইন বদলাবে
        dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

        P, G = [], []
        with torch.no_grad():
            for sil, meta, y in dl:
                x = to_input(sil.to(dev), meta.to(dev)).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(x)
                P.append((p.float() * y_std + y_mean).cpu())
                G.append((y.to(dev).float() * y_std + y_mean).cpu())
        P = torch.cat(P).numpy()
        G = torch.cat(G).numpy()
        err = np.abs(P - G)                  # cm, shape (N, 14)

        # per-photo CSV
        res = df.copy()
        for i, t in enumerate(TARGETS):
            res[f"gt_{t}"] = G[:, i]
            res[f"pred_{t}"] = P[:, i]
            res[f"err_{t}"] = err[:, i]
        res.to_csv(f"{out}/{split}_predictions.csv", index=False)

        # metrics: MAE + 90/75/50-th percentile of abs error (cm)
        tab = pd.DataFrame({
            "measurement": TARGETS,
            "MAE_cm": err.mean(0),
            "TP90_cm": np.percentile(err, 90, axis=0),
            "TP75_cm": np.percentile(err, 75, axis=0),
            "TP50_cm": np.percentile(err, 50, axis=0),
        })
        tab.loc[len(tab)] = ["MEAN", tab.MAE_cm.mean(), tab.TP90_cm.mean(),
                             tab.TP75_cm.mean(), tab.TP50_cm.mean()]
        tab.to_csv(f"{out}/{split}_metrics.csv", index=False)

        summary[split] = {"n_photos": int(len(df)), "mean_MAE_cm": float(err.mean())}
        print(f"\n===== {split}  ({len(df)} photos) =====")
        print(tab.round(3).to_string(index=False))

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    runs_vol.commit()
    print("\nSaved to", out)


@app.local_entrypoint()
def main(run: str = "bmnet_v2", ckpt: str = "best.pt"):
    evaluate.remote(run=run, ckpt=ckpt)