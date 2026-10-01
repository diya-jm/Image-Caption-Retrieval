"""Loading captions + precomputed VGG19 features, building the vocabulary, GloVe."""
import numpy as np
import torch
from collections import Counter

ROOT = "../data/coco"   # scripts are run from the code/ folder


def load_split(split, root=ROOT):
    """Return (features [N,4096], captions [5N], img_idx [5N]).
    Caption i belongs to image i // 5."""
    feats = np.load(f"{root}/images/10crop/{split}.npy").astype("float32")
    with open(f"{root}/{split}.txt") as f:
        caps = [line.strip() for line in f]
    assert len(caps) == 5 * len(feats), "expected 5 captions per image"
    img_idx = np.arange(len(caps)) // 5
    return feats, caps, img_idx


def tokenize(caption):
    return caption.lower().split()


def build_vocab(captions):
    """word -> id. 0 = <pad>, 1 = <unk>."""
    counts = Counter(w for c in captions for w in tokenize(c))
    word2idx = {"<pad>": 0, "<unk>": 1}
    for w in sorted(counts):
        word2idx[w] = len(word2idx)
    return word2idx


def encode(caption, word2idx):
    return [word2idx.get(w, 1) for w in tokenize(caption)]


def load_glove(path, word2idx, dim=300, seed=0):
    """Matrix [vocab, dim]. Words missing from GloVe get small random vectors."""
    rng = np.random.RandomState(seed)
    emb = rng.normal(0, 0.1, (len(word2idx), dim)).astype("float32")
    emb[0] = 0.0
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            if w in word2idx:
                emb[word2idx[w]] = np.asarray(parts[1:], dtype="float32")
                found += 1
    print(f"GloVe: found {found}/{len(word2idx)} vocabulary words")
    return emb


def pad_batch(id_lists):
    """List of id lists -> (padded LongTensor [B,T], lengths LongTensor [B])."""
    lengths = torch.tensor([max(1, len(x)) for x in id_lists])
    out = torch.zeros(len(id_lists), int(lengths.max()), dtype=torch.long)
    for i, x in enumerate(id_lists):
        if len(x):
            out[i, :len(x)] = torch.tensor(x)
    return out, lengths