"""Recall@K and mean rank, caption -> image retrieval."""
import torch


@torch.no_grad()
def recall_at_k(cap_emb, img_emb, caps_per_img=5, fold_size=1000, ks=(1, 5, 10), max_folds=None):
    """cap_emb: [caps_per_img*N, D] (captions of image j are rows j*c .. j*c+c-1)
    img_emb: [N, D]. Retrieval is done inside folds of `fold_size` images
    (1K images, as in the paper) and results are averaged over folds.
    Returns {"R@1":..., "R@5":..., "R@10":..., "mean_rank":...} (recalls in %)."""
    cap_emb, img_emb = torch.as_tensor(cap_emb), torch.as_tensor(img_emb)
    n_img = img_emb.shape[0]
    fold_size = min(fold_size, n_img)
    n_folds = n_img // fold_size
    if max_folds:
        n_folds = min(n_folds, max_folds)
    rows = {f"R@{k}": [] for k in ks}
    rows["mean_rank"] = []
    for f in range(n_folds):
        im = img_emb[f * fold_size:(f + 1) * fold_size]
        cp = cap_emb[f * fold_size * caps_per_img:(f + 1) * fold_size * caps_per_img]
        d = torch.cdist(cp.float(), im.float())                       # [caps, imgs]
        target = torch.arange(fold_size, device=d.device).repeat_interleave(caps_per_img)
        correct = d.gather(1, target[:, None])
        ranks = (d < correct).sum(1) + 1                               # 1 = best
        for k in ks:
            rows[f"R@{k}"].append((ranks <= k).float().mean().item() * 100)
        rows["mean_rank"].append(ranks.float().mean().item())
    return {k: sum(v) / len(v) for k, v in rows.items()}