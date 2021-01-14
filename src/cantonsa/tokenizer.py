from transformers import AutoTokenizer, BertTokenizer

import torch.nn as nn
import logging
from constants import SPEC_TOKEN

logger = logging.getLogger(__name__)


def get_tokenizer(source, name):
    """
    name (str): source:::tokenizer
    """
    logger.info("***** Loading tokenizer *****")
    logger.info("  Source = '%s'", source)
    logger.info("  Tokenizer = '%s'", name)
    
    if source == "transformers":
        # if "bert" in name.lower():
        #     tokenizer = BertTokenizer.from_pretrained(name, use_fast=True)
        # else:
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

        tokenizer.add_special_tokens({"additional_special_tokens": [SPEC_TOKEN.TARGET]})
        return tokenizer
    else:
        raise ("Tokenizer not found.")
