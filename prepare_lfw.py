# prepare_lfw.py
import os
import csv
import random
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people

# --------------------- Face detection (optional) ---------------------
# We use OpenCV Haar for a quick, lightweight crop. If cv2 isn't available,
# alignment will automatically be skipped.
def build_haar():
    try:
        import cv2
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return None

def crop_face_haar(pil_img, haar, target_size):
    """
    Returns a tightly cropped face (RGB) using Haar; if no face,
    returns a plain resized image (no alignment).
    """
    import cv2
    rgb = np.asarray(pil_img)                   # RGB uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return pil_img.resize(target_size, Image.BILINEAR)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = rgb[y:y+h, x:x+w, :]
    face = Image.fromarray(face, mode='RGB').resize(target_size, Image.BILINEAR)
    return face

# --------------------- Helpers ---------------------
def safe_id(name: str) -> str:
    return name.replace(" ", "_")

def ensure_uint8_rgb(arr):
    """
    Ensure an image array is RGB uint8 for saving.
    Handles float arrays in [0,1] or [0,255], and grayscale.
    """
    arr = np.asarray(arr)

    # Expand grayscale -> RGB
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)

    # Convert dtype/range
    if np.issubdtype(arr.dtype, np.floating):
        maxv = float(arr.max()) if arr.size else 1.0
        minv = float(arr.min()) if arr.size else 0.0

        if maxv <= 1.0 + 1e-6:
            # floats in [0,1]
            arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
        elif maxv <= 255.0 + 1e-6:
            # floats in [0,255]
            arr = arr.round().clip(0, 255).astype(np.uint8)
        else:
            # unexpected range -> min-max normalize
            arr = ((arr - minv) / (maxv - minv + 1e-12) * 255.0).round().clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


# --------------------- Main script ---------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare LFW: gallery/probe split for identification.")
    parser.add_argument("--out-dir", type=str, default="data", help="Output root directory.")
    parser.add_argument("--meta-dir", type=str, default="meta", help="Directory for metadata (CSV).")
    parser.add_argument("--img-size", type=int, default=224, help="Square output size (e.g., 112, 160, 224).")
    parser.add_argument("--min-images-per-id", type=int, default=3, help="Keep identities with at least this many images.")
    parser.add_argument("--gallery-per-id", type=int, default=1, help="How many reference images per identity in gallery.")
    parser.add_argument("--max-ids", type=int, default=0, help="If >0, cap the number of identities kept (for quick tests).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--align", choices=["none", "haar"], default="haar",
                        help="Face cropping method: 'haar' (recommended) or 'none' (just resize).")
    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    META_DIR = Path(args.meta_dir)
    IMG_SIZE = (args.img_size, args.img_size)
    GALLERY_PER_ID = args.gallery_per_id
    MIN_IMAGES_PER_ID = args.min_images_per_id
    RANDOM_SEED = args.seed
    MAX_IDS = args.max_ids

    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    # Download LFW (color=True gives RGB images)
    lfw = fetch_lfw_people(color=True, funneled=True, resize=1.0, download_if_missing=True)
    X = lfw.images                      # (N, h, w, 3) typically uint8 RGB
    names = lfw.target_names            # list of person names
    y = lfw.target                      # integer labels

    # Group indices by identity
    groups = defaultdict(list)
    for i, yi in enumerate(y):
        groups[names[yi]].append(i)

    # Filter IDs with enough images
    kept_items = [(pid, idxs) for pid, idxs in groups.items() if len(idxs) >= MIN_IMAGES_PER_ID]
    kept_items.sort(key=lambda kv: kv[0])  # deterministic order by name

    if MAX_IDS and MAX_IDS > 0:
        kept_items = kept_items[:MAX_IDS]

    # Prepare folders
    (OUT_DIR / "gallery").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe").mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Optional detector
    haar = build_haar() if args.align == "haar" else None
    if args.align == "haar" and haar is None:
        print("[WARN] OpenCV not available; proceeding without alignment.")
        args.align = "none"

    rows = []
    total_gallery = 0
    total_probe = 0

    print(f"Total IDs in LFW: {len(groups)} | Kept (>= {MIN_IMAGES_PER_ID} imgs): {len(kept_items)}"
          + (f" | Capped to first {MAX_IDS}" if MAX_IDS else ""))

    for pid, idxs in tqdm(kept_items, desc="Building split"):
        random.shuffle(idxs)
        gallery_idxs = idxs[:GALLERY_PER_ID]
        probe_idxs   = idxs[GALLERY_PER_ID:]

        gid = safe_id(pid)
        gdir = OUT_DIR / "gallery" / gid
        pdir = OUT_DIR / "probe" / gid
        gdir.mkdir(parents=True, exist_ok=True)
        pdir.mkdir(parents=True, exist_ok=True)

        def process_and_save(idx, split, count):
            # Ensure standard RGB uint8 image (no normalization)
            arr = ensure_uint8_rgb(X[idx])
            img = Image.fromarray(arr, mode='RGB')

            # Optional face crop/alignment (RGB in, RGB out)
            if args.align == "haar" and haar is not None:
                img = crop_face_haar(img, haar, IMG_SIZE)
            else:
                img = img.resize(IMG_SIZE, Image.BILINEAR)

            fn = f"{gid}_{count:04d}.jpg"
            out_path = (gdir if split == "gallery" else pdir) / fn
            # Save as standard 8-bit JPEG — will NOT look dark
            img.save(out_path, quality=95)
            rows.append([str(out_path), gid, split])

        # Save gallery images
        for k, idx in enumerate(gallery_idxs):
            process_and_save(idx, "gallery", k)
            total_gallery += 1

        # Save probe images
        for k, idx in enumerate(probe_idxs):
            process_and_save(idx, "probe", k)
            total_probe += 1

    # Write CSV mapping
    csv_path = META_DIR / "splits.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "identity", "split"])
        writer.writerows(rows)

    n_ids = len(kept_items)
    print(f"Done.\nIDs kept: {n_ids} | Gallery images: {total_gallery} | Probe images: {total_probe}")
    print(f"CSV: {csv_path}")
    print("Tip: If images ever look dark, it means normalized arrays got saved by mistake. "
          "This script always saves uint8 RGB, so you’re safe here.")

if __name__ == "__main__":
    main()
