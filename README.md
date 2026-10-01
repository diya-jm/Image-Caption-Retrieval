# Neural Caption-Image Retrieval

**UE24CS352A: Machine Learning, Mini-Project**

Type a sentence, get the matching photos. This project replicates the caption-to-image retrieval models from *"Neural Caption-Image Retrieval"* (Qian & Lamberti, Stanford CS229) and extends them with one additional model of our own.

**Team:** `<Name 1> (<SRN>)` and `<Name 2> (<SRN>)`
**Section:** `<section>`  |  **Faculty:** `<faculty name>`

---

## Overview

Given a text query such as *"a dog sitting on a couch"*, the system searches a database of 1,000 images and returns the ones that best match the description.

Both images and sentences are converted into vectors in a shared 1,024-dimensional space. The model is trained so that a photo and its own caption end up close together, and a photo and an unrelated caption end up far apart. At search time, the query sentence is embedded and the nearest images are returned.

```
photo  -> VGG19 features (4096) -> linear layer ----+
                                                    +--> shared space -> nearest neighbours
caption -> GloVe -> GRU / LSTM -> last state -------+
```

## Models

| Model | Description | Role |
|---|---|---|
| Baseline | Average of the caption's GloVe vectors, mapped to the image feature space with linear regression | Replication |
| Baseline + Weight | Same, but averaging the features of all 5 test captions of an image | Replication |
| GRU + GloVe | GRU caption encoder with GloVe-initialised, fine-tuned word vectors, trained with a margin ranking loss | Replication |
| LSTM + GloVe | Same, with an LSTM | Replication |
| Extra model | `<describe your model, e.g. Transformer caption encoder>` | Our extension |

**Loss:** pairwise hinge (margin) ranking loss with in-batch negatives, using the similarity S(i, c) = -||f_i(i) - f_c(c)||². Margin = 0.1.

## Dataset

- **MS COCO** (123,287 images, 5 human-written captions each), using the standard split of 113,287 train / 5,000 validation / 5,000 test images.
- Image features are the **precomputed 10-crop VGG19 fc7 features** (4096-d) released with the order-embeddings work.
- Word representations come from **GloVe** (6B tokens, 300-d).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── code/
│   ├── main.py                  # entry point: train a model
│   ├── trainer.py               # training loop, checkpointing, early stopping
│   ├── dataset.py               # data loading, vocabulary, GloVe
│   ├── models.py                # image encoder, GRU/LSTM caption encoder, ranking loss
│   ├── evaluation.py            # Recall@K and mean rank
│   ├── baseline.py              # averaged GloVe + linear regression baseline
│   ├── extra_model.py           # our additional model
│   ├── eval_only.py             # evaluate a saved model on the test set
│   └── eval_custom_input.py     # interactive demo: type a caption, get top-10 images
└── results/                     # training logs and result tables
```

## Setup

1. Clone the repository and install the requirements:

```
   git clone https://github.com/<username>/Image-Caption-Retrieval.git
   cd Image-Caption-Retrieval
   pip install -r requirements.txt
```

2. Download the dataset (about 1 GB) into the repository root and unzip it. This creates `data/coco/`:

```
   wget http://www.cs.toronto.edu/~vendrov/order/coco.zip
   unzip coco.zip
```

3. Download the GloVe vectors from https://nlp.stanford.edu/projects/glove/ (`glove.6B.zip`) and place `glove.6B.300d.txt` in `data/`.

A GPU is strongly recommended for training (we used Google Colab).

## Usage

Run everything from the `code/` folder.

```
cd code

# Baselines
python baseline.py --glove ../data/glove.6B.300d.txt

# Train the GRU and LSTM models
python main.py --rnn gru  --glove ../data/glove.6B.300d.txt
python main.py --rnn lstm --glove ../data/glove.6B.300d.txt

# Train our extra model
python main.py --rnn transformer --glove ../data/glove.6B.300d.txt

# Evaluate a saved model on the test set
python eval_only.py --ckpt ../checkpoints/gru_glove.pt

# Interactive demo
python eval_custom_input.py --ckpt ../checkpoints/gru_glove.pt
```

Useful training options: `--epochs`, `--batch`, `--lr`, `--margin`, `--dim`, `--patience`, `--ckpt` (where to save the best model). Run `python main.py --help` for the full list.

## Evaluation

We report **Recall@K** (R@K): the percentage of caption queries for which the correct image is among the top K retrieved images, and the **mean rank** of the correct image. Retrieval is performed over 1,000 images, and results are averaged over the five 1K folds of the test set.

## Results

Reported values are from the reference paper (Table 1). "Ours" columns are from our own runs.

| Method | R@1 (paper) | R@1 (ours) | R@10 (paper) | R@10 (ours) | Mean rank (paper) | Mean rank (ours) |
|---|---|---|---|---|---|---|
| Baseline | 10.3 | | 17.1 | | 176.1 | |
| Baseline + Weight | 19.8 | | 65.5 | | 10.5 | |
| GRU + GloVe | 37.0 | | 86.8 | | 7.3 | |
| LSTM + GloVe | 35.4 | | 86.2 | | 7.5 | |
| **Extra model** | n/a | | n/a | | n/a | |

*Discussion of differences between reported and obtained results: `<fill in>`*

## Our extension

`<Which model you added, why you chose it, and how it compares with the GRU/LSTM models.>`

## Differences from the reference

- `<e.g. learning rate, gradient clipping, random seeds, library versions>`

## References

1. J. Qian and G. Lamberti, *Neural Caption-Image Retrieval*, Stanford CS229 project. Code: https://github.com/giacomolamberti90/CS229_project
2. I. Vendrov, R. Kiros, S. Fidler, R. Urtasun, *Order-Embeddings of Images and Language*, arXiv:1511.06361, 2015.
3. T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
4. K. Simonyan and A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, arXiv:1409.1556, 2014.
5. J. Pennington, R. Socher, C. Manning, *GloVe: Global Vectors for Word Representation*, EMNLP 2014.
6. K. Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder*, arXiv:1406.1078, 2014.