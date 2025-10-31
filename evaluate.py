# evaluate.py — Complete version (English, with initials for confusion matrix)
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import OrderedDict


# ================== Helper Functions ==================
def to_initials(label: str) -> str:
    """
    Convert a label like 'First_Last' or 'John_Smith' to initials 'S.J'.
    Works well with LFW format (Firstname Lastname).
    """
    parts = re.split(r"[_\s\-]+", label.strip())
    parts = [p for p in parts if p]
    if not parts:
        return label[:1].upper()
    first = parts[0]
    last = parts[-1]
    f0 = first[0].upper() if first else ""
    l0 = last[0].upper() if last else ""
    return f"{l0}.{f0}"


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def rank1_predictions(probe_feats, gallery_feats, gallery_labels):
    sims = cosine_sim(probe_feats, gallery_feats)      # [P,G]
    idx = np.argmax(sims, axis=1)
    return np.array(gallery_labels)[idx], sims.max(axis=1), sims


def cmc_curve(probe_feats, probe_labels, gallery_feats, gallery_labels, max_rank=20):
    sims = cosine_sim(probe_feats, gallery_feats)      # [P,G]
    order = np.argsort(-sims, axis=1)
    ranked_labels = np.array(gallery_labels)[order]    # [P,G]
    ranks = np.full(len(probe_labels), 10**9, dtype=int)
    for i, y in enumerate(probe_labels):
        pos = np.where(ranked_labels[i] == y)[0]
        if len(pos):
            ranks[i] = pos[0]
    cmc = [np.mean(ranks < k) for k in range(1, max_rank + 1)]
    return np.array(cmc), ranks


def load_split(prefix):
    g = np.load(f"outputs/{prefix}_gallery.npz", allow_pickle=True)
    p = np.load(f"outputs/{prefix}_probe.npz", allow_pickle=True)
    return g["feats"], g["labels"], p["feats"], p["labels"]


def save_cmc(cmc_dict, out_png, out_csv=None):
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


def save_confusion(y_true, y_pred, labels, title, out_png,
                   normalize=True, display_labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels,
                          normalize="true" if normalize else None)
    tick_text = display_labels if display_labels is not None else labels
    fig_w = max(6, len(labels) * 0.4)
    fig_h = max(5, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues",
                   vmin=0.0 if normalize else None,
                   vmax=1.0 if normalize else None)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_text, rotation=90)
    ax.set_yticklabels(tick_text)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def print_summary(name, acc, cmc, ranks, file_handle=None):
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
    ap = argparse.ArgumentParser(description="Evaluate face identification models (print and save results).")
    ap.add_argument("--out-dir", type=str, default="evaluation",
                    help="Folder where results and plots will be saved.")
    ap.add_argument("--max-rank", type=int, default=20, help="Maximum rank for CMC curves.")
    ap.add_argument("--subset", type=int, default=20,
                    help="Number of identities to show in confusion matrices (to keep them readable).")
    ap.add_argument("--normalize-confusion", action="store_true",
                    help="Normalize confusion matrices by true class (recommended).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- FaceNet ----------
    gF, gL, pF, pL = load_split("facenet")
    yhat_fn, scores_fn, sims_fn = rank1_predictions(pF, gF, gL)
    acc_fn = (yhat_fn == pL).mean()
    cmc_fn, ranks_fn = cmc_curve(pF, pL, gF, gL, max_rank=args.max_rank)

    # ---------- ArcFace ----------
    gF2, gL2, pF2, pL2 = load_split("arcface")
    yhat_af, scores_af, sims_af = rank1_predictions(pF2, gF2, gL2)
    acc_af = (yhat_af == pL2).mean()
    cmc_af, ranks_af = cmc_curve(pF2, pL2, gF2, gL2, max_rank=args.max_rank)

    # ---------- Save text summary ----------
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        print_summary("FaceNet", acc_fn, cmc_fn, ranks_fn, file_handle=f)
        print_summary("ArcFace", acc_af, cmc_af, ranks_af, file_handle=f)

        better = "FaceNet" if acc_fn > acc_af else ("ArcFace" if acc_af > acc_fn else "Tie")
        comp = (
            "\n=== Comparison ===\n"
            f"  Higher Rank-1: {better}\n"
            f"  FaceNet  Rank-1: {acc_fn:.4f} | CMC@5: {cmc_fn[4] if len(cmc_fn)>=5 else np.nan:.4f}\n"
            f"  ArcFace  Rank-1: {acc_af:.4f} | CMC@5: {cmc_af[4] if len(cmc_af)>=5 else np.nan:.4f}\n"
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
    N = args.subset

    # FaceNet
    uniq_fn = sorted(list(set(pL.tolist())))
    subset_fn = uniq_fn[:N] if N > 0 else uniq_fn
    mask_fn = np.isin(pL, subset_fn)
    subset_fn_initials = [to_initials(x) for x in subset_fn]
    save_confusion(
        pL[mask_fn], yhat_fn[mask_fn], subset_fn,
        title="Confusion Matrix — FaceNet (subset)",
        out_png=os.path.join(args.out_dir, "confusion_facenet.png"),
        normalize=args.normalize_confusion,
        display_labels=subset_fn_initials
    )

    # ArcFace
    uniq_af = sorted(list(set(pL2.tolist())))
    subset_af = uniq_af[:N] if N > 0 else uniq_af
    mask_af = np.isin(pL2, subset_af)
    subset_af_initials = [to_initials(x) for x in subset_af]
    save_confusion(
        pL2[mask_af], yhat_af[mask_af], subset_af,
        title="Confusion Matrix — ArcFace (subset)",
        out_png=os.path.join(args.out_dir, "confusion_arcface.png"),
        normalize=args.normalize_confusion,
        display_labels=subset_af_initials
    )

    # ---------- Save label-initial mappings ----------
    with open(os.path.join(args.out_dir, "labels_initials_facenet.csv"), "w", encoding="utf-8") as f:
        f.write("original,initials\n")
        for o, i in zip(subset_fn, subset_fn_initials):
            f.write(f"{o},{i}\n")

    with open(os.path.join(args.out_dir, "labels_initials_arcface.csv"), "w", encoding="utf-8") as f:
        f.write("original,initials\n")
        for o, i in zip(subset_af, subset_af_initials):
            f.write(f"{o},{i}\n")

    # ---------- Display summary ----------
    print(open(summary_path, "r", encoding="utf-8").read())
    print(f"All results saved in: {os.path.abspath(args.out_dir)}")
