# extract_embeddings.py
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from models_facenet_arcface import FaceNetEmbedder, ArcFaceEmbedder

DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def iter_images(root: Path):
    # expects: root/ID_xxx/*.jpg
    for pid in sorted(os.listdir(root)):
        pdir = root / pid
        if not pdir.is_dir(): 
            continue
        for fn in sorted(os.listdir(pdir)):
            if fn.lower().endswith(('.jpg','.jpeg','.png')):
                yield pid, pdir / fn

def build_split(embedder, split_dir: Path):
    feats, labels, paths = [], [], []
    items = list(iter_images(split_dir))
    for pid, path in tqdm(items, desc=f"Embedding {split_dir.name}"):
        img = Image.open(path).convert('RGB')
        f = embedder.embed(img)
        feats.append(f); labels.append(pid); paths.append(str(path))
    return np.vstack(feats), np.array(labels), np.array(paths)

def run_one(backend_name: str):
    if backend_name == "facenet":
        emb = FaceNetEmbedder()
    elif backend_name == "arcface":
        emb = ArcFaceEmbedder()
    else:
        raise ValueError("backend_name must be 'facenet' or 'arcface'")

    gF, gL, gP = build_split(emb, DATA_DIR / "gallery")
    pF, pL, pP = build_split(emb, DATA_DIR / "probe")

    np.savez_compressed(OUT_DIR / f"{backend_name}_gallery.npz", feats=gF, labels=gL, paths=gP)
    np.savez_compressed(OUT_DIR / f"{backend_name}_probe.npz",  feats=pF, labels=pL, paths=pP)

if __name__ == "__main__":
    run_one("facenet")
    run_one("arcface")
    print("Done: embeddings saved in outputs/*.npz")
