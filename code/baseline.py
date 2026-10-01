"""Baseline: average GloVe vectors of the caption -> linear regression -> VGG feature space.
   python baseline.py --glove ../data/glove.6B.300d.txt"""
import argparse
import os
import numpy as np
import torch
from dataset import load_split, build_vocab, tokenize, load_glove
from evaluation import recall_at_k


def avg_glove(caps, word2idx, glove):
    out = np.zeros((len(caps), glove.shape[1]), dtype="float32")
    for i, c in enumerate(caps):
        ids = [word2idx.get(w, 1) for w in tokenize(c)]
        if ids:
            out[i] = glove[ids].mean(0)
    return out


def fit_linear(X, F, lam=1.0):
    """Closed-form ridge regression  W = argmin sum_k ||F[img(k)] - W^T x_k||^2 + lam||W||^2.
    X: [5N, 300] caption features (caption k belongs to image k//5), F: [N, 4096]."""
    Xb = np.hstack([X, np.ones((len(X), 1), dtype="float32")])          # add bias
    A = (Xb.T @ Xb).astype("float64")
    S = Xb.reshape(-1, 5, Xb.shape[1]).sum(1)                            # sum of the 5 captions per image
    B = (S.T @ F).astype("float64")
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), B).astype("float32")


def predict(X, W):
    return np.hstack([X, np.ones((len(X), 1), dtype="float32")]) @ W


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--glove", required=True)
    p.add_argument("--data", default="../data/coco")
    p.add_argument("--results", default="../results")
    p.add_argument("--lam", type=float, default=1.0)
    args = p.parse_args()

    f_tr, c_tr, _ = load_split("train", args.data)
    f_te, c_te, _ = load_split("test", args.data)
    word2idx = build_vocab(c_tr)
    glove = load_glove(args.glove, word2idx)
    W = fit_linear(avg_glove(c_tr, word2idx, glove), f_tr, args.lam)

    pred = torch.from_numpy(predict(avg_glove(c_te, word2idx, glove), W))
    feats = torch.from_numpy(f_te)
    r1 = recall_at_k(pred, feats)                                        # 1 caption per query
    pw = pred.reshape(-1, 5, pred.shape[1]).mean(1)                      # "Baseline + Weight": average the 5 captions
    r2 = recall_at_k(pw, feats, caps_per_img=1)
    lines = [f"Baseline        : R@1 {r1['R@1']:.1f} | R@10 {r1['R@10']:.1f} | mean rank {r1['mean_rank']:.1f}",
             f"Baseline+Weight : R@1 {r2['R@1']:.1f} | R@10 {r2['R@10']:.1f} | mean rank {r2['mean_rank']:.1f}"]
    print("\n".join(lines))
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "results_baseline.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")