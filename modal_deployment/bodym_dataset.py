"""BodyM data for BMnet: front+side silhouettes (640x480 each, side by side -> 640x960),
plus height/weight, 14 measurement targets (z-scored)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGETS = ["ankle", "arm-length", "bicep", "calf", "chest", "forearm", "height",
           "hip", "leg-length", "shoulder-breadth", "shoulder-to-crotch",
           "thigh", "waist", "wrist"]
VIEW_H, VIEW_W = 640, 480          # paper: each silhouette 640x480


def load_table(root, split):
    """One row per photo_id, with subject metadata and measurements joined."""
    d = Path(root) / split
    df = (pd.read_csv(d / "subject_to_photo_map.csv")
          .merge(pd.read_csv(d / "hwg_metadata.csv"), on="subject_id")
          .merge(pd.read_csv(d / "measurements.csv"), on="subject_id"))
    df["split"] = split
    return df.reset_index(drop=True)


def split_by_subject(df, val_frac=0.1, seed=0):
    """Validation split by SUBJECT (same person never in both train and val)."""
    subs = np.sort(df.subject_id.unique())
    rng = np.random.default_rng(seed)
    val = set(rng.choice(subs, max(1, int(round(len(subs) * val_frac))), replace=False))
    m = df.subject_id.isin(val)
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def make_stats(train_df):
    """Normalisation stats from training subjects only (one row per subject)."""
    s = train_df.drop_duplicates("subject_id")
    return {
        "y_mean": s[TARGETS].mean().values.astype(np.float32),
        "y_std": s[TARGETS].std().values.astype(np.float32),
        "hw_mean": s[["height_cm", "weight_kg"]].mean().values.astype(np.float32),
        "hw_std": s[["height_cm", "weight_kg"]].std().values.astype(np.float32),
    }


def _load_mask(path):
    im = Image.open(path).convert("L").resize((VIEW_W, VIEW_H), Image.BILINEAR)
    return (np.asarray(im) >= 128).astype(np.uint8)


class BodyM(Dataset):
    """Holds all masks in RAM as uint8 (about 3.8 GB for the full train split)."""

    def __init__(self, df, root, stats, workers=16):
        self.df, self.stats = df.reset_index(drop=True), stats
        root, n = Path(root), len(self.df)
        paths = [(root / s / "mask" / f"{p}.png",            # front
                  root / s / "mask_left" / f"{p}.png")       # side
                 for s, p in zip(self.df.split, self.df.photo_id)]
        self.masks = np.zeros((n, 2, VIEW_H, VIEW_W), np.uint8)

        def fill(i):
            self.masks[i, 0] = _load_mask(paths[i][0])
            self.masks[i, 1] = _load_mask(paths[i][1])

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(fill, range(n)))

        hw = self.df[["height_cm", "weight_kg"]].values.astype(np.float32)
        self.meta = ((hw - stats["hw_mean"]) / stats["hw_std"]).astype(np.float32)
        y = self.df[TARGETS].values.astype(np.float32)
        self.y = ((y - stats["y_mean"]) / stats["y_std"]).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sil = np.concatenate([self.masks[i, 0], self.masks[i, 1]], axis=1)  # (640, 960)
        return (torch.from_numpy(sil)[None],      # (1,640,960) uint8
                torch.from_numpy(self.meta[i]),   # (2,) normalised height, weight
                torch.from_numpy(self.y[i]))      # (14,) normalised targets


def to_input(sil, meta):
    """Paper input: silhouettes + constant height and weight channels -> (B,3,640,960).
    Call on the GPU inside the training loop."""
    b, _, h, w = sil.shape
    const = meta[:, :, None, None].expand(b, 2, h, w)
    return torch.cat([sil.float(), const], dim=1)


def denorm(y_z, stats):
    """Back to centimetres."""
    mean = torch.as_tensor(stats["y_mean"], device=y_z.device)
    std = torch.as_tensor(stats["y_std"], device=y_z.device)
    return y_z * std + mean