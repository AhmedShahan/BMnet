# weight_from_mesh.py  —  modal run weight_from_mesh.py 2>&1 | tee weight.txt
import modal

app = modal.App("nomo-weight")
image = modal.Image.debian_slim().pip_install("numpy", "pandas")
vol = modal.Volume.from_name("nomo")

DENSITY = 1.01   # kg per litre (assumption; volume is saved too, so you can change this later)


@app.function(image=image, cpu=8, memory=16384, timeout=3600, volumes={"/nomo": vol})
def run():
    import re, zipfile
    import numpy as np, pandas as pd

    z = zipfile.ZipFile("/nomo/smpl6890v_nicp_fits.zip")
    objs = [n for n in z.namelist() if n.endswith(".obj")]

    def parse(data):
        lines = data.decode().splitlines()
        v = np.array([l[2:].split() for l in lines if l.startswith("v ")], dtype=np.float64)
        f = np.array([[int(t.split("/")[0]) for t in l[2:].split()]
                      for l in lines if l.startswith("f ")], dtype=np.int64) - 1
        return v, f

    rows, f0, same_faces = [], None, 0
    for i, n in enumerate(objs):
        v, f = parse(z.read(n))
        if f0 is None:
            f0 = f
            # closed-manifold check on the template: every undirected edge used exactly twice
            e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
            _, cnt = np.unique(e, axis=0, return_counts=True)
            print(f"template: {len(v)} vertices, {len(f)} faces | edges used once: {(cnt==1).sum()}, "
                  f"twice: {(cnt==2).sum()}, >2: {(cnt>2).sum()}")
        same_faces += int(np.array_equal(f, f0))
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        signed = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0     # m^3
        sid = n.split("/")[-1][:-4]
        rows.append({"subject": sid, "gender": sid.split("_")[0],
                     "volume_L": abs(signed) * 1000.0, "vol_sign": np.sign(signed),
                     "mesh_height_m": v[:, 1].max() - v[:, 1].min()})
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(objs)} meshes")

    df = pd.DataFrame(rows)
    print(f"\nmeshes with identical face list to template: {same_faces}/{len(objs)}")
    print("negative-signed volumes (inward normals):", int((df.vol_sign < 0).sum()))

    # join with measurements (Head_Top_Height, mm) and image availability
    meas = pd.read_csv("/nomo/measurements.csv")[["subject", "Head_Top_Height"]]
    df = df.merge(meas, on="subject", how="left")
    zi = zipfile.ZipFile("/nomo/rendered_rgb.zip")
    img_ids = {m.group(1) for n in zi.namelist() for m in [re.search(r"((?:fe)?male_\d+)\.png$", n)] if m}
    df["has_mat"] = df.Head_Top_Height.notna()
    df["has_img"] = df.subject.isin(img_ids)
    df["has_all3"] = df.has_mat & df.has_img

    df["weight_kg"] = df.volume_L * DENSITY
    df["bmi_mesh"] = df.weight_kg / df.mesh_height_m ** 2
    df["height_diff_cm"] = (df.mesh_height_m * 1000 - df.Head_Top_Height) / 10.0
    df.drop(columns=["vol_sign"]).to_csv("/nomo/weight_estimates.csv", index=False)
    vol.commit()

    print(f"\nmeshes: {len(df)} | with mat: {int(df.has_mat.sum())} | with images: {int(df.has_img.sum())} "
          f"| ALL THREE: {int(df.has_all3.sum())}")
    print(df[df.has_all3].gender.value_counts().to_dict())

    def q(s):
        return s.describe(percentiles=[.05, .5, .95])[["min", "5%", "50%", "95%", "max"]].round(2)

    for g, d in df[df.has_all3].groupby("gender"):
        print(f"\n===== {g} (all three, n={len(d)}) =====")
        print(pd.DataFrame({"volume_L": q(d.volume_L), "weight_kg": q(d.weight_kg),
                            "mesh_height_m": q(d.mesh_height_m), "BMI": q(d.bmi_mesh)}).T.to_string())

    d = df[df.has_all3]
    print("\nmesh height minus Head_Top_Height (cm): median %.2f | p5 %.2f | p95 %.2f | |diff|>3cm: %d"
          % (d.height_diff_cm.median(), d.height_diff_cm.quantile(.05), d.height_diff_cm.quantile(.95),
             int((d.height_diff_cm.abs() > 3).sum())))
    print("BMI outside 15-45:", int(((d.bmi_mesh < 15) | (d.bmi_mesh > 45)).sum()), "of", len(d))
    print("\nsaved /nomo/weight_estimates.csv")


@app.local_entrypoint()
def main():
    run.remote()