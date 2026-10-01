"""Our own extra model (NOT in the reference paper).

TODO (Person B): implement TransformerCaptionEncoder so that it can replace the GRU:
    __init__(self, glove_matrix, dim)        # GloVe-initialised nn.Embedding + nn.TransformerEncoder
    forward(self, ids, lengths) -> [B, dim]  # one vector per caption (e.g. mean over non-pad tokens)
It must be an nn.Module. Everything else (image encoder, loss, evaluation) stays identical,
so the comparison with GRU/LSTM is fair.
Train it with:  python main.py --rnn transformer --glove ...
Then write down: why we chose it, and whether it is better/worse/similar than GRU/LSTM and why.
"""


class TransformerCaptionEncoder:
    def __init__(self, glove_matrix, dim):
        raise NotImplementedError("TODO: implement the extra model")