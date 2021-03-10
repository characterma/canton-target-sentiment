import logging
from transformers import AutoTokenizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from utils import SPEC_TOKEN

logger = logging.getLogger(__name__)


def get_tokenizer(source, name):
    """
    name (str): source:::tokenizer
    """
    logger.info("***** Loading tokenizer *****")
    logger.info("  Source = '%s'", source)
    logger.info("  Tokenizer = '%s'", name)

    if source == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        tokenizer.add_special_tokens({"additional_special_tokens": [SPEC_TOKEN.TARGET]})
        return tokenizer
    elif source == "spacy":
        if name == "english":
            nlp = English()
            tokenizer = Tokenizer(nlp.vocab)
            return tokenizer
        elif name == "chinese":
            return None
    else:
        return None
