# inspect_fits.py  —  modal run inspect_fits.py 2>&1 | tee inspect_fits.txt
import modal

app = modal.App("nomo-fits")
image = modal.Image.debian_slim().pip_install("scipy", "numpy").apt_install("curl")
vol = modal.Volume.from_name("nomo")

NAME = "smpl6890v_nicp_fits.zip"
URL = f"https://zenodo.org/records/5172126/files/{NAME}?download=1"


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes={"/nomo": vol})
def inspect_fits():
    import os, io, re, zipfile, subprocess, collections
    import numpy as np

    path = f"/nomo/{NAME}"
    if not os.path.exists(path):
        subprocess.run(["curl", "-L", "-f", "-o", path, URL], check=True)
        vol.commit()
    print(f"{NAME}: {os.path.getsize(path)/1e6:.1f} MB")

    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if not n.endswith("/")]
    print("files:", len(names))
    print("extensions:", dict(collections.Counter(os.path.splitext(n)[1].lower() for n in names)))
    pat = collections.Counter(re.sub(r"\d+", "N", n) for n in names)
    print("\npatterns:")
    for k, v in sorted(pat.items())[:20]:
        print(f"{v:6d}  {k}")
    print("\nfirst 15:")
    for n in names[:15]:
        print("  ", n)

    # inspect one sample
    n = names[0]
    ext = os.path.splitext(n)[1].lower()
    data = z.read(n)
    print(f"\n===== sample: {n}  ({len(data)/1e3:.1f} KB) =====")
    if ext == ".mat":
        from scipy.io import loadmat
        m = loadmat(io.BytesIO(data), squeeze_me=True, struct_as_record=False)
        for k, v in m.items():
            if not k.startswith("__"):
                print(" ", k, getattr(v, "shape", None), getattr(v, "dtype", None))
    elif ext in (".npy", ".npz"):
        a = np.load(io.BytesIO(data), allow_pickle=True)
        if ext == ".npz":
            for k in a.files:
                print(" ", k, a[k].shape, a[k].dtype)
        else:
            print(" ", a.shape, a.dtype)
    elif ext in (".obj", ".ply", ".txt"):
        txt = data.decode(errors="ignore").splitlines()
        print("\n".join(txt[:25]))
        print("lines:", len(txt),
              "| v:", sum(l.startswith("v ") for l in txt),
              "| f:", sum(l.startswith("f ") for l in txt))
    else:
        print(data[:200])


@app.local_entrypoint()
def main():
    inspect_fits.remote()