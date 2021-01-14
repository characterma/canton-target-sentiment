from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import XLNetConfig, BertConfig, ElectraConfig, XLMRobertaConfig, AlbertConfig
from transformers import (
    ElectraForPreTraining,
    ElectraModel,
    XLNetLMHeadModel,
    XLNetModel,
    XLMRobertaForMaskedLM,
    XLMRobertaModel,
    BertForMaskedLM,
    BertModel,
    AlbertModel
)
from utils import SPEC_TOKEN
import torch.nn as nn
import logging

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


class PretrainedLM(object):
    def __init__(self, model_name):
        """
        model_name:
            "toastynews/electra-hongkongese-large-discriminator"
            "toastynews/xlnet-hongkongese-base"
            "xlm-roberta-base"
            "xlm-roberta-large"
            "bert-base-multilingual-cased"
            "bert-base-chinese"
            "denpa92/bert-base-cantonese"
        """
        logger.info("***** Loading pretrained language model *****")
        logger.info("  Pretrained language model = '%s'", str(model_name))
        self.model_name = model_name
        self.config = CONFIG_CLASS_MAP[model_name].from_pretrained(model_name)
        self.model = MODEL_CLASS_MAP[model_name].from_pretrained(model_name)

    def resize_token_embeddings(self, tokenizer):
        self.model.resize_token_embeddings(len(tokenizer))
