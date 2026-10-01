"""Image encoder, caption encoder (GRU/LSTM + GloVe) and the ranking loss."""
import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """f_i(i) = W_i * f_VGG(i): linear map 4096 -> embedding dim."""
    def __init__(self, in_dim=4096, dim=1024):
        super().__init__()
        self.fc = nn.Linear(in_dim, dim)

    def forward(self, x):
        return self.fc(x)


class CaptionEncoder(nn.Module):
    """GloVe-initialised (fine-tuned) embeddings -> GRU/LSTM -> last hidden state."""
    def __init__(self, glove_matrix, dim=1024, rnn="gru"):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(
            torch.as_tensor(glove_matrix), freeze=False, padding_idx=0)
        cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn]
        self.rnn = cls(self.emb.embedding_dim, dim, batch_first=True)
        self.is_lstm = rnn == "lstm"

    def forward(self, ids, lengths):
        x = self.emb(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.is_lstm:
            h = h[0]
        return h[-1]                                                  # [B, dim]


def ranking_loss(cap, img, img_ids, margin=0.1):
    """Paper eq. (2) with S(i,c) = -||f_i(i) - f_c(c)||^2 and in-batch negatives.
    cap, img: [B, D] embeddings of matching pairs; img_ids: [B] image id of each pair
    (pairs sharing an image are not used as negatives of each other)."""
    d = (cap.unsqueeze(1) - img.unsqueeze(0)).pow(2).sum(2)           # d[i,j]=||cap_i-img_j||^2
    pos = d.diag().unsqueeze(1)                                       # [B,1]
    cost_img = (margin + pos - d).clamp(min=0)                        # wrong images for caption i
    cost_cap = (margin + pos - d.t()).clamp(min=0)                    # wrong captions for image i
    same = img_ids.unsqueeze(0) == img_ids.unsqueeze(1)
    mask = (~same).float()
    return ((cost_img + cost_cap) * mask).sum() / cap.shape[0]