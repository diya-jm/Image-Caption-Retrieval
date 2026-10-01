"""Evaluate a trained model on the test set (no training):
   python eval_only.py --ckpt ../checkpoints/gru_glove.pt"""
import argparse
import os
import torch
from dataset import load_split
from trainer import load_checkpoint, embed
from evaluation import recall_at_k

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--data", default="../data/coco")
p.add_argument("--results", default="../results")
p.add_argument("--split", default="test")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
img_enc, cap_enc, word2idx = load_checkpoint(args.ckpt, device)
feats, caps, _ = load_split(args.split, args.data)
cap_emb, img_emb = embed(img_enc, cap_enc, feats, caps, word2idx, device)
res = recall_at_k(cap_emb, img_emb)                              # average over 5 folds of 1K
name = os.path.splitext(os.path.basename(args.ckpt))[0]
line = (f"{name} [{args.split}, 1K images, avg of folds]: R@1 {res['R@1']:.1f} | "
        f"R@5 {res['R@5']:.1f} | R@10 {res['R@10']:.1f} | mean rank {res['mean_rank']:.1f}")
print(line)
os.makedirs(args.results, exist_ok=True)
with open(os.path.join(args.results, "test_results.txt"), "a") as f:
    f.write(line + "\n")