# test_nomo.py  —  modal run test_nomo.py 2>&1 | tee test_nomo.txt
import modal
from pathlib import Path

HERE = Path(__file__).parent.resolve()

app = modal.App("nomo-test")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy", "pandas", "pillow")
    .add_local_file(HERE / "bodym_dataset.py", "/root/bodym_dataset.py")
    .add_local_file(HERE / "bmnet.py", "/root/bmnet.py")
)
data_vol = modal.Volume.from_name("bodym")
nomo_vol = modal.Volume.from_name("nomo")
runs_vol = modal.Volume.from_name("bmnet-runs")

# BodyM target  ->  candidate NOMO columns (mm). Definitions are NOT guaranteed to match.
CAND = {
    "ankle": ["Ankle_Circ"],
    "arm-length": ["Shoulder_to_Wrist"],
    "bicep": ["Bicep_Circ"],
    "calf": ["Calf_Circ"],
    "chest": ["CHEST_MERGED"],                      # BUST_Circ (female) + CHEST_Circ (male)
    "height": ["Head_Top_Height"],
    "hip": ["HIP_Circ", "Hip_2_Circ", "TopHip_Circ"],
    "leg-length": ["Inseam", "Outseam"],
    "shoulder-breadth": ["Shoulder_to_Shoulder", "Across_Back"],
    "thigh": ["Thigh_Circ"],
    "waist": ["NaturalWAIST_Circ", "TrouserWAIST_Circ", "MaxWAIST_Circ"],
    "wrist": ["Wrist_Circ"],
}
CORE = [("chest", "CHEST_MERGED"), ("hip", "HIP_Circ"), ("waist", "NaturalWAIST_Circ"),
        ("thigh", "Thigh_Circ"), ("calf", "Calf_Circ"), ("bicep", "Bicep_Circ"),
        ("wrist", "Wrist_Circ"), ("ankle", "Ankle_Circ"), ("height", "Head_Top_Height")]


@app.function(image=image, gpu="L4", cpu=8, memory=32768, timeout=3600,
              volumes={"/data": data_vol, "/nomo": nomo_vol, "/runs": runs_vol})
def test(run="bmnet_v2"):
    import io, json, zipfile
    import numpy as np, pandas as pd, torch
    from PIL import Image
    from bodym_dataset import BodyM, load_table, to_input, TARGETS
    from bmnet import BMnet

    dev = "cuda"
    st = json.load(open(f"/runs/{run}/stats.json"))
    y_mean, y_std = np.array(st["y_mean"]), np.array(st["y_std"])
    hw_mean, hw_std = np.array(st["hw_mean"]), np.array(st["hw_std"])
    print("hw_mean (expect [cm, kg]):", hw_mean, "| hw_std:", hw_std)

    # mask value convention used by the dataset class (0/1 or 0/255)
    df0 = load_table("/data", "train").head(4).reset_index(drop=True)
    sil0 = BodyM(df0, "/data", st)[0][0]
    ONE = int(sil0.max())
    print("BodyM mask max value:", ONE, "| dtype:", sil0.dtype)

    model = BMnet().to(dev).to(memory_format=torch.channels_last)
    ck = torch.load(f"/runs/{run}/best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    # ---------- data ----------
    w = pd.read_csv("/nomo/weight_estimates.csv")
    w = w[w.has_all3].reset_index(drop=True)
    meas = pd.read_csv("/nomo/measurements.csv").set_index("subject")
    meas["CHEST_MERGED"] = meas["BUST_Circ"].fillna(meas["CHEST_Circ"])
    gt = meas.loc[w.subject]
    gender = w.gender.values
    N = len(w)
    print("subjects:", N)

    H, W = 640, 480

    def norm_view(a, th, tm):
        """crop to bbox, rescale so bbox height = th*H, place with top margin tm*H, centred."""
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
    print("preprocessed:", front.shape, side.shape)

    hcm = w.Head_Top_Height.values / 10.0
    Z = np.stack([(hcm - hw_mean[0]) / hw_std[0], (w.weight_kg.values - hw_mean[1]) / hw_std[1]], 1)

    def predict(flip):
        P = []
        for i in range(0, N, 64):
            f, s = front[i:i + 64], side[i:i + 64]
            if flip:
                s = s[:, :, ::-1]
            sil = torch.from_numpy(np.ascontiguousarray(np.concatenate([f, s], axis=2)[:, None])).to(dev)
            meta = torch.from_numpy(Z[i:i + 64]).float().to(dev)
            with torch.no_grad():
                x = to_input(sil, meta).contiguous(memory_format=torch.channels_last)
                p = model(x).float().cpu().numpy()
            P.append(p * y_std + y_mean)                # cm
        return np.concatenate(P)

    def report(P):
        rows = []
        for t, cols in CAND.items():
            j = TARGETS.index(t)
            for col in cols:
                g = gt[col].values / 10.0
                p = P[:, j]
                ok = np.isfinite(g) & (g > 0)
                lo, hi = np.percentile(g[ok], [0.5, 99.5])
                ok &= (g >= lo) & (g <= hi)             # drop sentinel / absurd values
                e = p[ok] - g[ok]; gm = gender[ok]
                rows.append(dict(target=t, gt_col=col, n=int(ok.sum()),
                                 MAE=np.abs(e).mean(), bias=e.mean(),
                                 MAE_debiased=np.abs(e - e.mean()).mean(),
                                 corr=np.corrcoef(p[ok], g[ok])[0, 1],
                                 MAE_F=np.abs(e[gm == "female"]).mean(),
                                 MAE_M=np.abs(e[gm == "male"]).mean()))
        return pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    P0 = predict(False)
    r0 = report(P0)
    print("\n===== side view as-is =====")
    print(r0.round(2).to_string(index=False))
    print("(MAE/bias in cm; bias = pred - NOMO; MAE_debiased = MAE after removing the mean offset;"
          " corr = Pearson r)")

    P1 = predict(True)
    r1 = report(P1)
    def core_mean(r):
        return np.mean([r[(r.target == t) & (r.gt_col == c)].MAE.iloc[0] for t, c in CORE])
    print(f"\ncore-set mean MAE (cm): side as-is = {core_mean(r0):.2f} | side mirrored = {core_mean(r1):.2f}")

    pred = pd.DataFrame(P0, columns=[f"pred_{t}" for t in TARGETS])
    pred.insert(0, "subject", w.subject.values)
    pred.to_csv("/nomo/nomo_predictions_bmnet.csv", index=False)
    nomo_vol.commit()
    print("saved /nomo/nomo_predictions_bmnet.csv")


@app.local_entrypoint()
def main():
    test.remote()