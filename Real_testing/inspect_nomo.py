# inspect_nomo.py  —  modal run inspect_nomo.py
import modal

app = modal.App("nomo-inspect")
image = modal.Image.debian_slim().pip_install("scipy", "numpy", "pillow").apt_install("curl")
vol = modal.Volume.from_name("nomo", create_if_missing=True)

BASE = "https://zenodo.org/records/5172126/files"


@app.function(image=image, cpu=4, memory=16384, timeout=7200, volumes={"/nomo": vol})
def inspect():
    import os, io, zipfile, subprocess, collections
    import numpy as np
    from scipy.io import loadmat
    from PIL import Image

    for name in ["bodymeasurement_mat.zip", "rendered_rgb.zip"]:
        path = f"/nomo/{name}"
        if not os.path.exists(path):
            subprocess.run(["curl", "-L", "-f", "-o", path, f"{BASE}/{name}?download=1"], check=True)
            vol.commit()
        print(f"\n===== {name}  ({os.path.getsize(path)/1e6:.1f} MB) =====")
        z = zipfile.ZipFile(path)
        names = z.namelist()
        print("entries:", len(names))
        ext = collections.Counter(os.path.splitext(n)[1].lower() for n in names)
        print("extensions:", dict(ext))
        print("first 25:"); [print("  ", n) for n in names[:25]]

        if name.startswith("bodymeasurement"):
            for n in names:
                if n.endswith(".mat"):
                    m = loadmat(io.BytesIO(z.read(n)))
                    print(f"\n{n}:")
                    for k, v in m.items():
                        if not k.startswith("__"):
                            print("  ", k, getattr(v, "shape", None), getattr(v, "dtype", None))
        else:
            imgs = [n for n in names if os.path.splitext(n)[1].lower() in (".png", ".jpg", ".jpeg")]
            for n in imgs[:4]:
                im = Image.open(io.BytesIO(z.read(n)))
                print(f"  {n}: size={im.size} mode={im.mode}")

    vol.commit()


@app.local_entrypoint()
def main():
    inspect.remote()