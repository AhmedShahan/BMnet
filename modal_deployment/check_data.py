from pathlib import Path
import modal

app = modal.App("bodym-check")
vol = modal.Volume.from_name("bodym")
image = (modal.Image.debian_slim()
         .pip_install("torch", "numpy", "pandas", "pillow",
                      extra_index_url="https://download.pytorch.org/whl/cpu")
         .add_local_python_source("bodym_dataset"))


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=1800)
def check():
    from torch.utils.data import DataLoader
    from bodym_dataset import (TARGETS, BodyM, load_table, make_stats,
                               split_by_subject, to_input)

    for split in ["train", "testA", "testB"]:
        df = load_table("/data", split)
        miss = sum(not (Path("/data") / split / d / f"{p}.png").exists()
                   for p in df.photo_id for d in ("mask", "mask_left"))
        nan = int(df[TARGETS + ["height_cm", "weight_kg"]].isna().sum().sum())
        print(f"{split}: {len(df)} photos, {df.subject_id.nunique()} subjects, "
              f"missing files {miss}, NaN {nan}")
        if split == "train":
            tr, va = split_by_subject(df)
            overlap = set(tr.subject_id) & set(va.subject_id)
            print(f"  split: train {len(tr)} photos / val {len(va)} photos, "
                  f"subject overlap {len(overlap)}")
            stats = make_stats(tr)
            print("  height/weight mean:", stats["hw_mean"], "std:", stats["hw_std"])
            print("  metadata height vs scan height, mean abs diff (cm):",
                  float((df.height_cm - df.height).abs().mean()))
            sample = tr.head(32)

    ds = BodyM(sample, "/data", stats)
    sil, meta, y = next(iter(DataLoader(ds, batch_size=8)))
    x = to_input(sil, meta)
    print("sil", tuple(sil.shape), sil.dtype, "values", sil.unique().tolist())
    print("model input", tuple(x.shape), x.dtype)
    print("foreground fraction", float(sil.float().mean()))
    print("targets", tuple(y.shape))


@app.local_entrypoint()
def main():
    check.remote()