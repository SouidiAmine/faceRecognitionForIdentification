# evaluate.py — Final version (top-30 confusion, numeric labels, fixed colorbar 0–6)
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import OrderedDict


# ================== Helper Functions ==================
def cosine_sim(a, b):
    """Compute cosine similarity between two feature matrices."""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def rank1_predictions(probe_feats, gallery_feats, gallery_labels):
    """Return predicted labels using highest cosine similarity."""
    sims = cosine_sim(probe_feats, gallery_feats)
    idx = np.argmax(sims, axis=1)
    return np.array(gallery_labels)[idx], sims.max(axis=1), sims


def cmc_curve(probe_feats, probe_labels, gallery_feats, gallery_labels, max_rank=20):
    """Compute CMC (Cumulative Match Characteristic) curve."""
    sims = cosine_sim(probe_feats, gallery_feats)
    order = np.argsort(-sims, axis=1)
    ranked_labels = np.array(gallery_labels)[order]
    ranks = np.full(len(probe_labels), 10**9, dtype=int)
    for i, y in enumerate(probe_labels):
        pos = np.where(ranked_labels[i] == y)[0]
        if len(pos):
            ranks[i] = pos[0]
    cmc = [np.mean(ranks < k) for k in range(1, max_rank + 1)]
    return np.array(cmc), ranks


def load_split(prefix, base_dir):
    """Load gallery and probe features + labels from specified folder."""
    g_path = os.path.join(base_dir, f"{prefix}_gallery.npz")
    p_path = os.path.join(base_dir, f"{prefix}_probe.npz")

    if not os.path.exists(g_path) or not os.path.exists(p_path):
        raise FileNotFoundError(f"[!] Missing files: {g_path} or {p_path}")

    g = np.load(g_path, allow_pickle=True)
    p = np.load(p_path, allow_pickle=True)
    return g["feats"], g["labels"], p["feats"], p["labels"]


def save_cmc(cmc_dict, out_png, out_csv=None):
    """Save CMC curves and optionally export as CSV."""
    plt.figure(figsize=(7, 5))
    for name, cmc in cmc_dict.items():
        ranks = np.arange(1, len(cmc) + 1)
        plt.plot(ranks, cmc, label=name)
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.title("Cumulative Match Characteristic (CMC) Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    if out_csv:
        rows = []
        for name, cmc in cmc_dict.items():
            for r, v in enumerate(cmc, start=1):
                rows.append(f"{r},{name},{v:.6f}")
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("rank,model,value\n")
            f.write("\n".join(rows))


def save_confusion_visual(cm, labels, title, out_png, top_k=30, normalize=True):
    """
    Visualize only the top-K most confused classes with fixed colorbar range
    and display numeric values inside each cell.
    """
    errors = np.sum(cm, axis=1) - np.diag(cm)
    top_idx = np.argsort(-errors)[:min(top_k, len(labels))]
    cm_subset = cm[np.ix_(top_idx, top_idx)]
    n_labels = len(top_idx)

    # === Normalize or fix range ===
    if normalize:
        row_sums = cm_subset.sum(axis=1, keepdims=True) + 1e-12
        cm_subset = cm_subset / row_sums
        vmin_value, vmax_value = 0.0, 1.0
    else:
        vmin_value, vmax_value = 0.0, 6.0  # fixed range

    fig, ax = plt.subplots(figsize=(max(6, n_labels * 0.4), max(5, n_labels * 0.4)))
    im = ax.imshow(cm_subset, interpolation="nearest", cmap="Blues",
                   vmin=vmin_value, vmax=vmax_value)

    # Colorbar
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{title} (Top {n_labels} confused classes)")
    ax.set_xlabel("Predicted index")
    ax.set_ylabel("True index")
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(np.arange(n_labels))
    ax.set_yticklabels(np.arange(n_labels))

    # === Add numeric values ===
    fmt = ".2f" if normalize else "d"
    thresh = (vmax_value + vmin_value) / 2.0
    for i in range(n_labels):
        for j in range(n_labels):
            val = format(cm_subset[i, j], fmt)
            ax.text(j, i, val,
                    ha="center", va="center",
                    color="white" if cm_subset[i, j] > thresh / 2 else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def print_summary(name, acc, cmc, ranks, file_handle=None):
    """Print and optionally save performance summary."""
    cmc1 = float(cmc[0]) if len(cmc) >= 1 else np.nan
    cmc5 = float(cmc[4]) if len(cmc) >= 5 else np.nan
    cmc10 = float(cmc[9]) if len(cmc) >= 10 else np.nan
    mean_rank = float(np.mean(ranks))
    median_rank = float(np.median(ranks))
    msg = (
        f"\n[{name}]\n"
        f"  Rank-1 accuracy : {acc:.4f}\n"
        f"  CMC@1 / @5 / @10: {cmc1:.4f} / {cmc5:.4f} / {cmc10:.4f}\n"
        f"  Mean rank       : {mean_rank:.2f}\n"
        f"  Median rank     : {median_rank:.2f}\n"
    )
    print(msg, end="")
    if file_handle:
        file_handle.write(msg)


# ================== MAIN SCRIPT ==================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate face identification models (top-30 confusion with color scale 0–6).")
    ap.add_argument("--data-dir", type=str, default="outputs",
                    help="Folder containing .npz files (default: outputs/)")
    ap.add_argument("--out-dir", type=str, default="evaluation",
                    help="Folder where results and plots will be saved.")
    ap.add_argument("--max-rank", type=int, default=20)
    ap.add_argument("--normalize-confusion", action="store_true")
    ap.add_argument("--top-k", type=int, default=30,
                    help="Number of most confused classes to display in the matrix.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- FaceNet ----------
    gF, gL, pF, pL = load_split("facenet", args.data_dir)
    yhat_fn, _, _ = rank1_predictions(pF, gF, gL)
    acc_fn = (yhat_fn == pL).mean()
    cmc_fn, ranks_fn = cmc_curve(pF, pL, gF, gL, max_rank=args.max_rank)

    # ---------- ArcFace ----------
    gF2, gL2, pF2, pL2 = load_split("arcface", args.data_dir)
    yhat_af, _, _ = rank1_predictions(pF2, gF2, gL2)
    acc_af = (yhat_af == pL2).mean()
    cmc_af, ranks_af = cmc_curve(pF2, pL2, gF2, gL2, max_rank=args.max_rank)

    # ---------- Save summary ----------
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        print_summary("FaceNet", acc_fn, cmc_fn, ranks_fn, file_handle=f)
        print_summary("ArcFace", acc_af, cmc_af, ranks_af, file_handle=f)
        better = "FaceNet" if acc_fn > acc_af else ("ArcFace" if acc_af > acc_fn else "Tie")
        comp = (
            "\n=== Comparison ===\n"
            f"  Higher Rank-1: {better}\n"
            f"  FaceNet  Rank-1: {acc_fn:.4f}\n"
            f"  ArcFace  Rank-1: {acc_af:.4f}\n"
        )
        print(comp, end="")
        f.write(comp)

    # ---------- Save CMC Curves ----------
    save_cmc(
        OrderedDict({"FaceNet": cmc_fn, "ArcFace": cmc_af}),
        out_png=os.path.join(args.out_dir, "cmc.png"),
        out_csv=os.path.join(args.out_dir, "cmc.csv"),
    )

    # ---------- Confusion Matrices ----------
    uniq_fn = sorted(list(set(pL.tolist())))
    cm_fn = confusion_matrix(pL, yhat_fn, labels=uniq_fn)
    np.save(os.path.join(args.out_dir, "confusion_facenet_global.npy"), cm_fn)
    save_confusion_visual(
        cm_fn, uniq_fn,
        title="Confusion Matrix — FaceNet",
        out_png=os.path.join(args.out_dir, "confusion_facenet_top30.png"),
        top_k=args.top_k,
        normalize=args.normalize_confusion
    )

    uniq_af = sorted(list(set(pL2.tolist())))
    cm_af = confusion_matrix(pL2, yhat_af, labels=uniq_af)
    np.save(os.path.join(args.out_dir, "confusion_arcface_global.npy"), cm_af)
    save_confusion_visual(
        cm_af, uniq_af,
        title="Confusion Matrix — ArcFace",
        out_png=os.path.join(args.out_dir, "confusion_arcface_top30.png"),
        top_k=args.top_k,
        normalize=args.normalize_confusion
    )

    print(f"\n[✔] All results saved in: {os.path.abspath(args.out_dir)}")
