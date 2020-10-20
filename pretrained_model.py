from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers import XLNetConfig, BertConfig, ElectraConfig, XLMRobertaConfig
from transformers import ElectraForPreTraining, ElectraModel, XLNetLMHeadModel, XLNetModel, XLMRobertaForMaskedLM, XLMRobertaModel, BertForMaskedLM, BertModel
import torch.nn as nn


CONFIG_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": ElectraConfig, 
    "toastynews/xlnet-hongkongese-base": XLNetConfig, 
    "xlm-roberta-base": XLMRobertaConfig, 
    "xlm-roberta-large": XLMRobertaConfig, 
    "bert-base-multilingual-cased": BertConfig, 
    "bert-base-chinese": BertConfig, 
    "denpa92/bert-base-cantonese": BertConfig, 
}

MODEL_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": ElectraModel, # ElectraForPreTraining
    "toastynews/xlnet-hongkongese-base": XLNetModel, # XLNetLMHeadModel
    "xlm-roberta-base": XLMRobertaModel, # XLMRobertaForMaskedLM
    "xlm-roberta-large": XLMRobertaModel, # XLMRobertaForMaskedLM
    "bert-base-multilingual-cased": BertModel, # BertForMaskedLM
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
        
    def to(self, device):
        self.model.to(device)


        

    

