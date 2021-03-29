

import unicodedata
import re
import logging
import collections
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
    elif source == "internal":
        return tokenizer_internal
    else:
        return None
        


def tokenizer_internal(text):
    Token = collections.namedtuple('Token', ['type', 'text', 'start', 'end'])

    token_specification = [
        ('REPLY', r'(引用(?:.|\n)*?發表)|(回覆樓主:)|(回覆(?:.|\n)*?帖子)'),
        ('URL', r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
                r'|(?:[a-zA-Z]|[0-9]|[.]|[!*\(\),])+\.[a-zA-Z]{2,}(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]'
                r'[0-9a-fA-F]))*'),
        ('EMAIL', r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z0-9]+"),  #
        ('ACCOUNT', r"@[^“”？@,?!。、，\n\xa0\t\r:\#\" ]+"),
        ("DIGIT", r"(?:\d+[,.]?\d*)"),
                ("LETTERS", r"Dr\."
                            '|'
                            "dr\."
                            "|"
                            "mrs?\."
                            "|"
                            "Mrs?\."
                            "|"
                            "[a-zA-Zàâçéèêëîïôûùüÿñæœ\.\'-\?]*[a-zA-Zàâçéèêëîïôûùüÿñæœ]"),
        ("EMOTION", r"(\[\S+?\])|(\(emoji\))"),
        ("QUESTION", r"([？?][”\"]?[\s]*)+"),
        ("PUNCTUATION", r"([!\n\r。！？?\.][”\"]?[\s]*)+"),
        ('SPACE', r'[ ]+'),
        ('CHAR', '[\u4E00-\u9FA5]'),  # Chinese character
        ('COMMA', r'[,；;，]+'),  # Any other character
        ('SPECIAL', r'.'),  # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    WORD_RE = re.compile(tok_regex, re.VERBOSE|re.IGNORECASE)

    def _tokenize(code):
        for mo in WORD_RE.finditer(code):
            kind = mo.lastgroup
            value = mo.group()
            start = mo.start()
            end = mo.end()
            yield Token(kind, value, start, end)

    return [token for token in _tokenize(text)]
