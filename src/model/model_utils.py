import logging
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import (
    XLNetConfig,
    BertConfig,
    ElectraConfig,
    XLMRobertaConfig,
    AlbertConfig,
)
from transformers import (
    ElectraForPreTraining,
    ElectraModel,
    XLNetLMHeadModel,
    XLNetModel,
    XLMRobertaForMaskedLM,
    XLMRobertaModel,
    BertForMaskedLM,
    BertModel,
    AlbertModel,
)

logger = logging.getLogger(__name__)

CONFIG_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": ElectraConfig,
    "toastynews/xlnet-hongkongese-base": XLNetConfig,
    "xlnet-base-cased": XLNetConfig,
    "xlm-roberta-base": XLMRobertaConfig,
    "xlm-roberta-large": XLMRobertaConfig,
    "bert-large-cased": BertConfig,
    "bert-base-cased": BertConfig,  # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    "bert-base-uncased": BertConfig,
    "bert-base-multilingual-cased": BertConfig,  # tested
    "bert-base-multilingual-uncased": BertConfig,
    "bert-base-chinese": BertConfig,  # testing
    "denpa92/bert-base-cantonese": BertConfig,  # missing tokenizer.
    "voidful/albert_chinese_tiny": AlbertConfig,  #
    "clue/albert_chinese_tiny": AlbertConfig,  #
    "voidful/albert_chinese_small": AlbertConfig,  #
    "clue/albert_chinese_small": AlbertConfig,
    "voidful/albert_chinese_base": AlbertConfig,  #
}

MODEL_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": ElectraModel,  # ElectraForPreTraining
    "toastynews/xlnet-hongkongese-base": XLNetModel,  # XLNetLMHeadModel
    "xlnet-base-cased": XLNetModel,
    "xlm-roberta-base": XLMRobertaModel,  # XLMRobertaForMaskedLM
    "xlm-roberta-large": XLMRobertaModel,  # XLMRobertaForMaskedLM
    "bert-large-cased": BertModel,
    "bert-base-cased": BertModel,
    "bert-base-uncased": BertModel,
    "bert-base-multilingual-cased": BertModel,  # BertForMaskedLM
    "bert-base-multilingual-uncased": BertModel,
    "bert-base-chinese": BertModel,  # BertForMaskedLM
    "denpa92/bert-base-cantonese": AlbertModel,
    "voidful/albert_chinese_tiny": AlbertModel,  #
    "clue/albert_chinese_tiny": AlbertModel,  #
    "voidful/albert_chinese_small": AlbertModel,  #
    "clue/albert_chinese_small": AlbertModel,
    "voidful/albert_chinese_base": AlbertModel,  #
}


def load_pretrained_bert(model_name):
    logger.info("***** Loading pretrained language model *****")
    logger.info("  Pretrained BERT = '%s'", str(model_name))
    return MODEL_CLASS_MAP[model_name].from_pretrained(model_name)


def load_pretrained_config(model_name):
    return CONFIG_CLASS_MAP[model_name].from_pretrained(model_name)


def load_pretrained_emb(emb_path):
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
