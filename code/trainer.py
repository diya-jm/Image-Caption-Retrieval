"""Training loop for the caption-image retrieval model (GRU/LSTM + GloVe)."""
import os
import random
import time

import numpy as np
import torch

from dataset import load_split, build_vocab, encode, load_glove, pad_batch
from models import ImageEncoder, CaptionEncoder, ranking_loss
from evaluation import recall_at_k


def make_cap_encoder(rnn, glove, dim):
    if rnn in ("gru", "lstm"):
        return CaptionEncoder(glove, dim, rnn)
    if rnn == "transformer":                       # our own extension model
        from extra_model import TransformerCaptionEncoder
        return TransformerCaptionEncoder(glove, dim)
    raise ValueError(f"unknown encoder type: {rnn}")


@torch.no_grad()
def embed(img_enc, cap_enc, feats, caps, word2idx, device, bs=1000):
    """Embed all images and captions of a split."""
    img_enc.eval()
    cap_enc.eval()
    img_emb = torch.cat([img_enc(torch.from_numpy(feats[i:i + bs]).to(device))
                         for i in range(0, len(feats), bs)])
    ids = [encode(c, word2idx) for c in caps]
    cap_emb = []
    for i in range(0, len(ids), bs):
        x, l = pad_batch(ids[i:i + bs])
        cap_emb.append(cap_enc(x.to(device), l))
    return torch.cat(cap_emb), img_emb


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device)
    vocab = len(ck["word2idx"])
    dummy = np.zeros((vocab, ck["glove_dim"]), dtype="float32")
    img_enc = ImageEncoder(4096, ck["dim"]).to(device)
    cap_enc = make_cap_encoder(ck["rnn"], dummy, ck["dim"]).to(device)
    img_enc.load_state_dict(ck["img"])
    cap_enc.load_state_dict(ck["cap"])
    return img_enc, cap_enc, ck["word2idx"]


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or f"{args.rnn}_glove"
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    log_path = os.path.join(args.results, f"results_{name}.txt")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    feats_tr, caps_tr, idx_tr = load_split("train", args.data)
    feats_va, caps_va, _ = load_split("val", args.data)
    word2idx = build_vocab(caps_tr)
    glove = load_glove(args.glove, word2idx)
    ids_tr = [encode(c, word2idx) for c in caps_tr]
    feats_tr_t = torch.from_numpy(feats_tr)

    img_enc = ImageEncoder(4096, args.dim).to(device)
    cap_enc = make_cap_encoder(args.rnn, glove, args.dim).to(device)
    params = list(img_enc.parameters()) + list(cap_enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    log(f"== {name} | {vars(args)} | device={device}")

    n, best, bad = len(ids_tr), -1.0, 0
    for epoch in range(1, args.epochs + 1):
        img_enc.train()
        cap_enc.train()
        perm = np.random.permutation(n)
        total, steps, t0 = 0.0, 0, time.time()
        for s in range(0, n - args.batch + 1, args.batch):
            b = perm[s:s + args.batch]
            x, l = pad_batch([ids_tr[j] for j in b])
            img_ids = torch.from_numpy(idx_tr[b]).to(device)
            im = img_enc(feats_tr_t[idx_tr[b]].to(device))
            cp = cap_enc(x.to(device), l)
            loss = ranking_loss(cp, im, img_ids, args.margin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)   # not in the paper, keeps training stable
            opt.step()
            total += loss.item()
            steps += 1
        cap_emb, img_emb = embed(img_enc, cap_enc, feats_va, caps_va, word2idx, device)
        res = recall_at_k(cap_emb, img_emb, max_folds=1)          # first 1K val images
        log(f"epoch {epoch:3d} loss {total / steps:.4f} | val R@1 {res['R@1']:.1f} "
            f"R@5 {res['R@5']:.1f} R@10 {res['R@10']:.1f} mean_r {res['mean_rank']:.1f} "
            f"| {time.time() - t0:.0f}s")
        if res["R@10"] > best:
            best, bad = res["R@10"], 0
            torch.save({"img": img_enc.state_dict(), "cap": cap_enc.state_dict(),
                        "word2idx": word2idx, "dim": args.dim, "rnn": args.rnn,
                        "glove_dim": glove.shape[1]}, args.ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stopping (no val R@10 improvement for {args.patience} epochs)")
                break
    log(f"best val R@10: {best:.1f} (checkpoint: {args.ckpt})")