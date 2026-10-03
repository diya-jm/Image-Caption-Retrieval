"""Own extra model (NOT in the reference paper): Transformer caption encoder.

Same inputs/outputs as the GRU/LSTM encoder (word ids -> one vector per caption),
so the image encoder, loss, data and evaluation are identical and the comparison is fair.
Train with:  python main.py --rnn transformer --glove ../data/glove.6B.300d.txt --lr 0.0001
"""
import torch
import torch.nn as nn


class TransformerCaptionEncoder(nn.Module):
    def __init__(self, glove_matrix, dim=1024, d_model=512, nhead=8, layers=2,
                 ff=1024, dropout=0.1, max_len=128):
        super().__init__()
        # same GloVe initialisation (fine-tuned) as the GRU/LSTM models
        self.emb = nn.Embedding.from_pretrained(
            torch.as_tensor(glove_matrix), freeze=False, padding_idx=0)
        self.proj = nn.Linear(self.emb.embedding_dim, d_model)   # 300 -> d_model
        self.pos = nn.Embedding(max_len, d_model)                # learned position vectors
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, dim)                       # d_model -> shared space

    def forward(self, ids, lengths):
        B, T = ids.shape
        idx = torch.arange(T, device=ids.device)
        valid = idx.unsqueeze(0) < lengths.to(ids.device).unsqueeze(1)   # True = real word
        x = self.proj(self.emb(ids)) + self.pos(idx.clamp(max=self.max_len - 1))
        x = self.encoder(x, src_key_padding_mask=~valid)         # words attend to each other
        m = valid.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1)                       # average over real words only
        return self.out(pooled)                                  # [B, dim]