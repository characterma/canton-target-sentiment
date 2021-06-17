import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm 


logger = logging.getLogger(__name__)


def _load_pretrained_emb(emb_path):
    logger.info("***** Loading pretrained embeddings *****")
    vectors = []
    with open(emb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            break
        for line in tqdm(f):
            vectors.append(line.rstrip().split(' ')[1:])
    vectors = np.array(vectors, dtype=float)
    logger.info("  Embeddings size = '%s'", str(vectors.shape))
    return vectors


class WordEmbeddings(nn.Module):
    def __init__(self, pretrained_emb_path=None, embedding_trainable=False, emb_dim=None, vocab_size=None, emb_dropout=0):
        super(WordEmbeddings, self).__init__()
        if pretrained_emb_path is not None:
            embeddings = _load_pretrained_emb(pretrained_emb_path)
            _, emb_dim = embeddings.shape
            embeddings = np.concatenate([np.zeros([1, emb_dim]), embeddings], axis=0) 
            self.embed = nn.Embedding.from_pretrained(
                torch.tensor(embeddings),
                freeze=(not embedding_trainable),
            )
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.emb_dim = emb_dim
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.embed(x)
        x = self.emb_dropout(x)
        return x
