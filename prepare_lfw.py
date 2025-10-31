# prepare_lfw.py
import os, random, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people

OUT_DIR = Path("data")
GALLERY_PER_ID = 1         # images de référence par identité
MIN_IMAGES_PER_ID = 3      # garder seulement les identités avec >=3 images
RANDOM_SEED = 42
IMG_SIZE = (224, 224)      # taille uniforme

def safe_id(name: str) -> str:
    return name.replace(" ", "_")

def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    lfw = fetch_lfw_people(color=True, funneled=True, resize=1.0, download_if_missing=True)
    X = lfw.images                      # (N, h, w, 3) uint8
    names = lfw.target_names
    y = lfw.target

    # Grouper indices par identité
    groups = defaultdict(list)
    for i, yi in enumerate(y):
        groups[names[yi]].append(i)

    kept = {pid: idxs for pid, idxs in groups.items() if len(idxs) >= MIN_IMAGES_PER_ID}
    (OUT_DIR / "gallery").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe").mkdir(parents=True, exist_ok=True)
    Path("meta").mkdir(exist_ok=True)

    rows = []
    for pid, idxs in tqdm(kept.items(), desc="Building split"):
        random.shuffle(idxs)
        gallery_idxs = idxs[:GALLERY_PER_ID]
        probe_idxs   = idxs[GALLERY_PER_ID:]

        gid = safe_id(pid)
        gdir = OUT_DIR / "gallery" / gid
        pdir = OUT_DIR / "probe" / gid
        gdir.mkdir(parents=True, exist_ok=True)
        pdir.mkdir(parents=True, exist_ok=True)

        def save(idx, split, count):
            img = Image.fromarray(X[idx].astype("uint8")).resize(IMG_SIZE, Image.BILINEAR)
            fn = f"{gid}_{count:04d}.jpg"
            out = (gdir if split == "gallery" else pdir) / fn
            img.save(out, quality=95)
            rows.append([str(out), gid, split])

        for k, idx in enumerate(gallery_idxs): save(idx, "gallery", k)
        for k, idx in enumerate(probe_idxs):   save(idx, "probe", k)

    with open("meta/splits.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["path","identity","split"], *rows])

    n_ids = len(kept)
    n_gallery = sum(1 for _,_,s in rows if s=="gallery")
    n_probe  = sum(1 for _,_,s in rows if s=="probe")
    print(f"Done. IDs: {n_ids} | Gallery images: {n_gallery} | Probe images: {n_probe}")

if __name__ == "__main__":
    main()
