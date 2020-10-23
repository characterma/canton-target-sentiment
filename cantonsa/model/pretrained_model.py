from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import XLNetConfig, BertConfig, ElectraConfig, XLMRobertaConfig
from transformers import ElectraForPreTraining, ElectraModel, XLNetLMHeadModel, XLNetModel, XLMRobertaForMaskedLM, XLMRobertaModel, BertForMaskedLM, BertModel
import torch.nn as nn


CONFIG_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": ElectraConfig, 
    "toastynews/xlnet-hongkongese-base": XLNetConfig, 
    "xlnet-base-cased": XLNetConfig, 
    "xlm-roberta-base": XLMRobertaConfig, 
    "xlm-roberta-large": XLMRobertaConfig, 
    "bert-large-cased": BertConfig, 
    "bert-base-cased": BertConfig,  # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    "bert-base-multilingual-cased": BertConfig,  # tested
    "bert-base-multilingual-uncased": BertConfig, 
    "bert-base-chinese": BertConfig,  # testing
    "denpa92/bert-base-cantonese": BertConfig,  # missing tokenizer.
}

MODEL_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": ElectraModel, # ElectraForPreTraining
    "toastynews/xlnet-hongkongese-base": XLNetModel, # XLNetLMHeadModel
    "xlnet-base-cased": XLNetModel, 
    "xlm-roberta-base": XLMRobertaModel, # XLMRobertaForMaskedLM
    "xlm-roberta-large": XLMRobertaModel, # XLMRobertaForMaskedLM
    "bert-large-cased": BertModel, 
    "bert-base-cased": BertModel, 
    "bert-base-multilingual-cased": BertModel, # BertForMaskedLM
    "bert-base-multilingual-uncased": BertModel, 
    "bert-base-chinese": BertModel, # BertForMaskedLM
    "denpa92/bert-base-cantonese": BertModel
}

class PretrainedML(object):
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
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.config = CONFIG_CLASS_MAP[model_name].from_pretrained(model_name)
        print(self.config)
        self.model = MODEL_CLASS_MAP[model_name].from_pretrained(model_name)
        
    # def to(self, device):
    #     self.model.to(device)


        

    

