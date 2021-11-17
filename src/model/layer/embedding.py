import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


def load_word_embeddings(emb_path, word_to_id):
    logger.info("***** Loading pretrained embeddings *****")
    
    emb_dim = None
    with open(emb_path, encoding="utf-8", errors="ignore") as f:
        for _ in f:
            break
        for line in tqdm(f):
            line = line.rstrip().split(" ")
            emb_dim = len(line[1:])
            break

    embeddings = np.zeros((len(word_to_id), emb_dim))
    
    with open(emb_path, encoding="utf-8", errors="ignore") as f:
        for _ in f:
            break
        for line in tqdm(f):
            line = line.rstrip().split(" ")
            word = line[0]

            if word in word_to_id:
                word_id = word_to_id[word]
                embeddings[word_id] = line[1:]
            
    logger.info("  Embeddings size = '%s'", str(embeddings.shape))
    return embeddings


class WordEmbeddings(nn.Module):
    def __init__(
        self,
        pretrained_emb_path=None,
        embedding_trainable=False,
        emb_dim=None,
        vocab_size=None,
        emb_dropout=0,
        word_to_id=None
    ):
        super(WordEmbeddings, self).__init__()
        if pretrained_emb_path is not None:
            embeddings = load_word_embeddings(pretrained_emb_path, word_to_id)
            _, emb_dim = embeddings.shape
            embeddings = torch.tensor(embeddings).float()
            self.embed = nn.Embedding.from_pretrained(
                embeddings, freeze=(not embedding_trainable)
            )
        else:
            self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.emb_dim = emb_dim
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.embed(x)
        x = self.emb_dropout(x)
        return x
