# compare_sil.py  —  modal run compare_sil.py 2>&1 | tee compare.txt
import modal
from pathlib import Path

HERE = Path(__file__).parent.resolve()

app = modal.App("sil-compare")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy", "pandas", "pillow")
    .add_local_file(HERE / "bodym_dataset.py", "/root/bodym_dataset.py")
    .add_local_file(HERE / "bmnet.py", "/root/bmnet.py")
)
data_vol = modal.Volume.from_name("bodym")
nomo_vol = modal.Volume.from_name("nomo")
runs_vol = modal.Volume.from_name("bmnet-runs")


def bbox(m):
    H, W = m.shape
    r = np.where(m.any(1))[0]; c = np.where(m.any(0))[0]
    return dict(top=r[0] / H, bottom=(H - 1 - r[-1]) / H, h=(r[-1] - r[0] + 1) / H,
                w=(c[-1] - c[0] + 1) / W, cx=(c[0] + c[-1]) / 2 / W, fg=m.mean())


import numpy as np


@app.function(image=image, cpu=4, memory=16384, timeout=1800,
              volumes={"/data": data_vol, "/nomo": nomo_vol, "/runs": runs_vol})
def compare(n=300, run="bmnet_v2"):
    import io, json, zipfile
    import pandas as pd
    from PIL import Image
    from bodym_dataset import BodyM, load_table

    st = json.load(open(f"/runs/{run}/stats.json"))
    hw_mean, hw_std = np.array(st["hw_mean"]), np.array(st["hw_std"])

    # ---------- BodyM (training distribution) ----------
    df = load_table("/data", "train").reset_index(drop=True).sample(n, random_state=0).reset_index(drop=True)
    ds = BodyM(df, "/data", st)
    rows, ex_b = [], []
    for i in range(len(ds)):
        sil, meta, _ = ds[i]
        sil = sil[0].numpy() > 0                       # (640, 960)
        hcm = float(meta[0]) * hw_std[0] + hw_mean[0]
        for view, m in (("front", sil[:, :480]), ("side", sil[:, 480:])):
            rows.append({"set": "BodyM", "view": view, "height_cm": hcm, **bbox(m)})
        if i < 4:
            ex_b.append((sil[:, :480], sil[:, 480:]))

    # ---------- NOMO ----------
    w = pd.read_csv("/nomo/weight_estimates.csv")
    w = w[w.has_all3].sample(n, random_state=0)
    zi = zipfile.ZipFile("/nomo/rendered_rgb.zip")
    ex_n, vals = [], set()
    for k, (sid, g, hmm) in enumerate(zip(w.subject, w.gender, w.Head_Top_Height)):
        pair = []
        for view in ("front", "side"):
            a = np.array(Image.open(io.BytesIO(zi.read(f"rendered_rgb/{g}/{view}/silh/{sid}.png"))))
            vals.update(np.unique(a)[:5].tolist())
            m = a > 127
            rows.append({"set": "NOMO", "view": view, "height_cm": hmm / 10.0, **bbox(m)})
            pair.append(m)
        if k < 4:
            ex_n.append(tuple(pair))
    print("NOMO silh pixel values (first few unique):", sorted(vals)[:10])

    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for (s, v), d in r.groupby(["set", "view"]):
        print(f"\n===== {s} / {v} (n={len(d)}) =====")
        print(d[["top", "bottom", "h", "w", "cx", "fg"]].describe(percentiles=[.05, .5, .95])
              .loc[["min", "5%", "50%", "95%", "max"]].round(3).to_string())
        print("corr(bbox height fraction, person height):", round(d.h.corr(d.height_cm), 3))

    # ---------- montage ----------
    def tile(m, H=320):
        im = Image.fromarray((m * 255).astype(np.uint8)).convert("L")
        return im.resize((int(im.width * H / im.height), H))

    rowsimg = []
    for exs in (ex_b, ex_n):
        tiles = [tile(x) for pair in exs for x in pair]
        Wt = sum(t.width + 8 for t in tiles)
        canvas = Image.new("L", (Wt, 320), 90)
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0)); x += t.width + 8
        rowsimg.append(canvas)
    out = Image.new("L", (max(c.width for c in rowsimg), 660), 90)
    out.paste(rowsimg[0], (0, 0)); out.paste(rowsimg[1], (0, 340))
    out.save("/nomo/compare.png")
    nomo_vol.commit()
    print("\nsaved /nomo/compare.png  (top row: BodyM, bottom row: NOMO; each person = front, side)")


@app.local_entrypoint()
def main():
    compare.remote()