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
    vectors = []
    with open(emb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            break
        for line in tqdm(f):
            vectors.append(line.rstrip().split(' ')[1:])
    return np.array(vectors, dtype=float)


class BaseModel(nn.Module):
    INPUT = []

    def __init__(self):
        super(BaseModel, self).__init__()
        pass

    def load_state(self, state_path=None, state_dict=None):
        if state_path is not None:
            if "*." in state_path.name:
                file_ext = "." + state_path.name.split(".")[-1]
                for f in list(state_path.parent.glob("*"))[
                    -1::-1
                ]:  # use the last saved model
                    if f.name.endswith(file_ext):
                        state_path = f
                        break
            logger.info("***** Loading model state *****")
            logger.info("  Path = %s", str(state_path))
            assert state_path.is_file()
            state_dict = torch.load(state_path, map_location="cpu")
        elif state_dict is not None:
            pass
        else:
            return
        self.load_state_dict(state_dict)
        self.to(self.device)