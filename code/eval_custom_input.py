"""Interactive demo: type a caption, get the top-10 matching test images.
   python eval_custom_input.py --ckpt ../checkpoints/gru_glove.pt
Only image numbers are printed. TODO (demo): to *show* the pictures, add the raw COCO
photos and map the test-image index to the right file."""
import argparse
import torch
from dataset import load_split, encode, pad_batch
from trainer import load_checkpoint, embed

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--data", default="../data/coco")
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
img_enc, cap_enc, word2idx = load_checkpoint(args.ckpt, device)
feats, caps, _ = load_split("test", args.data)
_, img_emb = embed(img_enc, cap_enc, feats[:1000], caps[:5], word2idx, device)  # first 1K test images

while True:
    try:
        q = input("caption (empty to quit): ").strip()
    except EOFError:
        break
    if not q:
        break
    x, l = pad_batch([encode(q, word2idx)])
    with torch.no_grad():
        cap_enc.eval()
        c = cap_enc(x.to(device), l)
        d = torch.cdist(c, img_emb)[0]
    top = d.topk(10, largest=False)
    print("top-10 image indices:", top.indices.tolist())