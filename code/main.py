"""Train a model:  python main.py --rnn gru --glove ../data/glove.6B.300d.txt"""
import argparse
from trainer import train

p = argparse.ArgumentParser()
p.add_argument("--rnn", default="gru", choices=["gru", "lstm", "transformer"])
p.add_argument("--glove", required=True, help="path to glove.6B.300d.txt")
p.add_argument("--data", default="../data/coco")
p.add_argument("--results", default="../results")
p.add_argument("--ckpt", default=None, help="where to save the best model (use a Drive path in Colab)")
p.add_argument("--name", default=None)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=0.001, help="paper reports 0.05; try both and report")
p.add_argument("--margin", type=float, default=0.1)
p.add_argument("--dim", type=int, default=1024)
p.add_argument("--patience", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
if args.ckpt is None:
    args.ckpt = f"../checkpoints/{args.name or args.rnn + '_glove'}.pt"
train(args)