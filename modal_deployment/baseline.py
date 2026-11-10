# baseline.py  —  modal run --detach baseline.py --run bmnet_v2
# height + weight  ->  14 measurements, ordinary linear regression (no silhouettes)
import modal
from pathlib import Path

HERE = Path(__file__).parent.resolve()

app = modal.App("bmnet-baseline")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "pandas", "pillow")
    .add_local_file(HERE / "bodym_dataset.py", "/root/bodym_dataset.py")
    .add_local_file(HERE / "bmnet.py", "/root/bmnet.py")
)
data_vol = modal.Volume.from_name("bodym")
runs_vol = modal.Volume.from_name("bmnet-runs")


@app.function(image=image, cpu=8, memory=32768, timeout=7200,
              volumes={"/data": data_vol, "/runs": runs_vol})
def baseline(run="bmnet_v2", workers=6):
    import os, json
    import numpy as np, pandas as pd, torch
    from torch.utils.data import DataLoader
    from bodym_dataset import BodyM, load_table, TARGETS

    out = f"/runs/{run}/baseline"
    os.makedirs(out, exist_ok=True)
    st = json.load(open(f"/runs/{run}/stats.json"))
    y_mean = np.array(st["y_mean"], dtype=np.float64)
    y_std = np.array(st["y_std"], dtype=np.float64)

    def collect(split):
        df = load_table("/data", split).reset_index(drop=True)
        ds = BodyM(df, "/data", st)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers)
        X, Y = [], []
        for sil, meta, y in dl:              # meta = z-scored [height, weight]
            X.append(meta.numpy()); Y.append(y.numpy())
        return df, np.concatenate(X).astype(np.float64), np.concatenate(Y).astype(np.float64)

    # fit on the train split (z-space, with intercept)
    _, Xtr, Ytr = collect("train")
    A = np.c_[Xtr, np.ones(len(Xtr))]
    W, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    print("train rows:", len(Xtr))

    summary = {}
    for split in ["testA", "testB"]:
        df, X, Yz = collect(split)
        Pz = np.c_[X, np.ones(len(X))] @ W
        P = Pz * y_std + y_mean
        G = Yz * y_std + y_mean
        err = np.abs(P - G)                  # cm

        res = df.copy()
        for i, t in enumerate(TARGETS):
            res[f"gt_{t}"] = G[:, i]; res[f"pred_{t}"] = P[:, i]; res[f"err_{t}"] = err[:, i]
        res.to_csv(f"{out}/{split}_predictions.csv", index=False)

        tab = pd.DataFrame({
            "measurement": TARGETS,
            "MAE_cm": err.mean(0),
            "TP90_cm": np.percentile(err, 90, axis=0),
            "TP75_cm": np.percentile(err, 75, axis=0),
            "TP50_cm": np.percentile(err, 50, axis=0),
        })
        tab.loc[len(tab)] = ["MEAN", tab.MAE_cm.mean(), tab.TP90_cm.mean(),
                             tab.TP75_cm.mean(), tab.TP50_cm.mean()]
        flat = err.reshape(-1)
        tab.loc[len(tab)] = ["POOLED(paper)", flat.mean(), np.percentile(flat, 90),
                             np.percentile(flat, 75), np.percentile(flat, 50)]

        # side by side with BMnet (fp32 eval)
        bm_path = f"/runs/{run}/eval_fp32/{split}_metrics.csv"
        if os.path.exists(bm_path):
            bm = pd.read_csv(bm_path).set_index("measurement")
            tab["BMnet_MAE_cm"] = tab["measurement"].map(bm["MAE_cm"])
            tab["BMnet_gain_%"] = 100 * (1 - tab["BMnet_MAE_cm"] / tab["MAE_cm"])
        tab.to_csv(f"{out}/{split}_metrics.csv", index=False)

        summary[split] = {"n_photos": int(len(df)), "mean_MAE_cm": float(err.mean()),
                          "pooled_TP90_cm": float(np.percentile(flat, 90)),
                          "pooled_TP75_cm": float(np.percentile(flat, 75)),
                          "pooled_TP50_cm": float(np.percentile(flat, 50))}
        print(f"\n===== baseline {split}  ({len(df)} photos) =====")
        print(tab.round(3).to_string(index=False))

    json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
    runs_vol.commit()
    print("\nSaved to", out)


@app.local_entrypoint()
def main(run: str = "bmnet_v2"):
    baseline.remote(run=run)