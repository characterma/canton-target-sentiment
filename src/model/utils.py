import logging
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utils import load_config
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


def load_pretrained_bert(model_config):
    logger.info("***** Loading pretrained language model *****")
    prev_model_dir = model_config.get('pretrained_lm_from_prev', None)
    if prev_model_dir is None:
        # 
        model_name = model_config['pretrained_lm']
        model_dir = model_config.get('pretrained_lm_dir', model_name)
        logger.info("  Pretrained BERT = '%s'", str(model_dir))
        return MODEL_CLASS_MAP[model_name].from_pretrained(model_dir)
    else:
        prev_args = get_args(prev_model_dir)
        prev_args = load_config(prev_args)
        model_path = prev_model_dir / "model.pt"
        model = get_model(model_path)
        return model.pretrained_model


def load_pretrained_config(model_config):
    prev_model_dir = model_config.get('pretrained_lm_from_prev', None)
    if prev_model_dir is None:
        model_name = model_config['pretrained_lm']
        model_dir = model_config.get('pretrained_lm_dir', model_name)
        return CONFIG_CLASS_MAP[model_name].from_pretrained(model_dir)
    else:
        prev_args = get_args(prev_model_dir)
        prev_args = load_config(prev_args)
        model_name = prev_args.model_config['pretrained_lm']
        return CONFIG_CLASS_MAP[model_name].from_pretrained(model_name)

