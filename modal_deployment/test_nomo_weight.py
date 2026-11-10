# test_nomo_weight.py  —  modal run test_nomo_weight.py 2>&1 | tee weight_sens.txt
import modal
from pathlib import Path

HERE = Path(__file__).parent.resolve()

app = modal.App("nomo-weight-sens")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "pandas", "pillow")
    .add_local_file(HERE / "bodym_dataset.py", "/root/bodym_dataset.py")
    .add_local_file(HERE / "bmnet.py", "/root/bmnet.py")
)
data_vol = modal.Volume.from_name("bodym")
nomo_vol = modal.Volume.from_name("nomo")
runs_vol = modal.Volume.from_name("bmnet-runs")

CORE = [("chest", "CHEST_MERGED"), ("hip", "HIP_Circ"), ("waist", "TrouserWAIST_Circ"),
        ("thigh", "Thigh_Circ"), ("calf", "Calf_Circ"), ("bicep", "Bicep_Circ"),
        ("wrist", "Wrist_Circ"), ("height", "Head_Top_Height")]


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3600,
              volumes={"/data": data_vol, "/nomo": nomo_vol, "/runs": runs_vol})
def sens(run="bmnet_v2"):
    import io, json, zipfile
    import numpy as np, pandas as pd, torch
    from PIL import Image
    from bodym_dataset import BodyM, load_table, to_input, TARGETS
    from bmnet import BMnet

    dev = "cuda"
    st = json.load(open(f"/runs/{run}/stats.json"))
    y_mean, y_std = np.array(st["y_mean"]), np.array(st["y_std"])
    hw_mean, hw_std = np.array(st["hw_mean"]), np.array(st["hw_std"])

    df0 = load_table("/data", "train").head(4).reset_index(drop=True)
    ONE = int(BodyM(df0, "/data", st)[0][0].max())

    model = BMnet().to(dev).to(memory_format=torch.channels_last)
    ck = torch.load(f"/runs/{run}/best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    w = pd.read_csv("/nomo/weight_estimates.csv")
    w = w[w.has_all3].reset_index(drop=True)
    meas = pd.read_csv("/nomo/measurements.csv").set_index("subject")
    meas["CHEST_MERGED"] = meas["BUST_Circ"].fillna(meas["CHEST_Circ"])
    gt = meas.loc[w.subject]
    N = len(w)

    H, W = 640, 480

    def norm_view(a, th, tm):
        m = a > 127
        r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
        crop = a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
        s = th * H / crop.shape[0]
        nw, nh = max(1, round(crop.shape[1] * s)), round(crop.shape[0] * s)
        arr = np.array(Image.fromarray(crop).resize((nw, nh), Image.BILINEAR)) >= 128
        out = np.zeros((H, W), np.uint8)
        y0, x0 = round(tm * H), (W - nw) // 2
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + nh), min(W, x0 + nw)
        out[ys:ye, xs:xe] = arr[ys - y0:ye - y0, xs - x0:xe - x0]
        return out * ONE

    zi = zipfile.ZipFile("/nomo/rendered_rgb.zip")

    def read(g, view, sid):
        return np.array(Image.open(io.BytesIO(zi.read(f"rendered_rgb/{g}/{view}/silh/{sid}.png"))))

    front = np.stack([norm_view(read(g, "front", s), 0.862, 0.067) for s, g in zip(w.subject, w.gender)])
    side = np.stack([norm_view(read(g, "side", s), 0.895, 0.061) for s, g in zip(w.subject, w.gender)])

    hcm = w.Head_Top_Height.values / 10.0
    wkg = w.weight_kg.values

    def predict(weight):
        Z = np.stack([(hcm - hw_mean[0]) / hw_std[0], (weight - hw_mean[1]) / hw_std[1]], 1)
        P = []
        for i in range(0, N, 64):
            sil = torch.from_numpy(np.ascontiguousarray(
                np.concatenate([front[i:i + 64], side[i:i + 64]], axis=2)[:, None])).to(dev)
            meta = torch.from_numpy(Z[i:i + 64]).float().to(dev)
            with torch.no_grad():
                x = to_input(sil, meta).contiguous(memory_format=torch.channels_last)
                p = model(x).float().cpu().numpy()
            P.append(p * y_std + y_mean)
        return np.concatenate(P)

    variants = {
        "-10%": wkg * 0.90, "-5%": wkg * 0.95, "est(1.00)": wkg,
        "+5%": wkg * 1.05, "+10%": wkg * 1.10,
        "mean(%.0fkg)" % hw_mean[1]: np.full(N, hw_mean[1]),
    }
    P = {k: predict(v) for k, v in variants.items()}
    base = P["est(1.00)"]

    # valid-GT masks (same cleaning as before)
    masks = {}
    for t, col in CORE:
        g = gt[col].values / 10.0
        ok = np.isfinite(g) & (g > 0)
        lo, hi = np.percentile(g[ok], [0.5, 99.5])
        masks[(t, col)] = (ok & (g >= lo) & (g <= hi), g)

    mae = {}
    for k, Pk in P.items():
        col_vals = []
        for t, col in CORE:
            ok, g = masks[(t, col)]
            col_vals.append(np.abs(Pk[ok, TARGETS.index(t)] - g[ok]).mean())
        mae[k] = col_vals
    tab = pd.DataFrame(mae, index=[f"{t} ({c})" for t, c in CORE])
    tab.loc["MEAN"] = tab.mean()
    pd.set_option("display.width", 220)
    print("weight deviation vs estimate (mean |dw| kg):",
          {k: round(float(np.abs(v - wkg).mean()), 1) for k, v in variants.items()})
    print("\n===== MAE (cm) vs NOMO, by weight variant =====")
    print(tab.round(2).to_string())

    print("\n===== change in MAE vs est(1.00) (cm; + = worse) =====")
    print((tab.sub(tab["est(1.00)"], axis=0)).round(2).to_string())

    shift = pd.DataFrame({k: (P[k] - base).mean(0) for k in ["-10%", "+10%", "mean(%.0fkg)" % hw_mean[1]]},
                         index=TARGETS)
    shift["sd_+10%"] = (P["+10%"] - base).std(0)
    print("\n===== mean prediction shift vs est(1.00) for ALL 14 outputs (cm; no ground truth needed) =====")
    print(shift.round(2).to_string())


@app.local_entrypoint()
def main():
    sens.remote()