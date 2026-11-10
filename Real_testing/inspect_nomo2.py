# inspect_nomo2.py  —  modal run inspect_nomo2.py 2>&1 | tee inspect2.txt
import modal

app = modal.App("nomo-inspect2")
image = modal.Image.debian_slim().pip_install("scipy", "numpy", "pandas", "pillow")
vol = modal.Volume.from_name("nomo")


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/nomo": vol})
def inspect2():
    import io, re, zipfile, collections
    import numpy as np, pandas as pd
    from scipy.io import loadmat
    from PIL import Image

    # ---------- 1. measurements ----------
    z = zipfile.ZipFile("/nomo/bodymeasurement_mat.zip")
    rows = []
    for n in z.namelist():
        if not n.endswith(".mat"):
            continue
        sid = n.split("/")[-1][:-4]                       # e.g. female_1538
        m = loadmat(io.BytesIO(z.read(n)), squeeze_me=True, struct_as_record=False)
        s = m["s"]
        r = {"subject": sid, "gender": sid.split("_")[0]}
        for f in s._fieldnames:
            v = np.asarray(getattr(s, f)).squeeze()
            r[f] = float(v) if v.size == 1 else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("/nomo/measurements.csv", index=False)
    vol.commit()

    print("subjects:", len(df), df.gender.value_counts().to_dict())
    cols = [c for c in df.columns if c not in ("subject", "gender")]
    st = pd.DataFrame({
        "n_valid": df[cols].notna().sum(),
        "min": df[cols].min(), "p5": df[cols].quantile(0.05),
        "median": df[cols].median(),
        "p95": df[cols].quantile(0.95), "max": df[cols].max(),
    })
    print("\n===== measurement values (as stored) =====")
    print(st.round(2).to_string())
    print("\nfields valid for ALL subjects:", int((st.n_valid == len(df)).sum()), "of", len(cols))

    # ---------- 2. rendered images ----------
    z2 = zipfile.ZipFile("/nomo/rendered_rgb.zip")
    names = [n for n in z2.namelist() if not n.endswith("/")]
    print(f"\n===== rendered_rgb.zip : {len(names)} files =====")
    pat = collections.Counter(re.sub(r"\d+", "N", n) for n in names)
    for k, v in sorted(pat.items()):
        print(f"{v:6d}  {k}")

    folders = collections.Counter("/".join(n.split("/")[:-1]) for n in names)
    print("\nfolders:")
    for k, v in sorted(folders.items()):
        print(f"{v:6d}  {k}")

    # one sample image size per folder
    print("\nsample sizes:")
    seen = set()
    for n in names:
        d = "/".join(n.split("/")[:-1])
        if d in seen or not n.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        seen.add(d)
        im = Image.open(io.BytesIO(z2.read(n)))
        print(f"  {d}: {n.split('/')[-1]}  size={im.size} mode={im.mode}")

    # overlap of subject ids between measurements and images
    img_ids = {re.search(r"((?:fe)?male_\d+)", n).group(1) for n in names if re.search(r"((?:fe)?male_\d+)", n)}
    mat_ids = set(df.subject)
    print(f"\nimage subjects: {len(img_ids)} | mat subjects: {len(mat_ids)} | in both: {len(img_ids & mat_ids)}")


@app.local_entrypoint()
def main():
    inspect2.remote()