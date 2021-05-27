

import unicodedata
import re
import logging
import collections
from transformers import AutoTokenizer, BertTokenizer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from utils import SPEC_TOKEN


logger = logging.getLogger(__name__)

TOKENIZER_CLASS_MAP = {
    "voidful/albert_chinese_tiny": BertTokenizer
}


def get_tokenizer(args, word_to_idx=None, required_token_types=None):
    # add special tokens

    source = args.model_config['tokenizer_source']
    name = args.model_config['tokenizer_name']
    logger.info("***** Loading tokenizer *****")
    logger.info("  Tokenizer source = '%s'", source)
    logger.info("  Tokenizer name = '%s'", name)
    if source == "transformers":
        Tokenizer = TOKENIZER_CLASS_MAP.get(name, AutoTokenizer)
        tokenizer = Tokenizer.from_pretrained(name, use_fast=True)
        return tokenizer
    elif source == "internal":
        return InternalTokenizer(word_to_idx=word_to_idx, required_token_types=required_token_types)
    else:
        raise ValueError("Unsupported tokenizer source.")


class TokensEncoded:
    # Interface class
    def __init__(self, tokens, char_to_token_dict, input_ids=None, attention_mask=None, token_type_ids=None):
        self._char_to_token_dict = char_to_token_dict
        self.tokens = tokens 
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def char_to_token(self, char_idx):
        return self._char_to_token_dict.get(char_idx, None)


class InternalTokenizer:
    def __init__(self, word_to_idx=None, required_token_types=None):
        self.word_to_idx = None
        self.idx_to_word = None
        if word_to_idx is not None:
            self.update_word_idx(word_to_idx)
        self.required_token_types = ["LETTERS", "CHAR", "DIGIT"] if required_token_types is None else required_token_types

    def update_word_idx(self, word_to_idx):
        self.word_to_idx = word_to_idx
        self.idx_to_word = dict()
        for k, v in word_to_idx.items():
            self.idx_to_word[v] = k

    def __call__(self, raw_text, max_length=None, truncation=True, add_special_tokens=True):
        token_spec = [
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
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_spec)
        WORD_RE = re.compile(tok_regex, re.VERBOSE|re.IGNORECASE)

        tokens = []
        char_to_token_dict = dict()
        input_ids = []
        attention_mask = []
        token_type_ids = []
        
        cur_token_idx = 0
        for mo in WORD_RE.finditer(raw_text):
            if max_length is None or len(tokens) < max_length:
                token_type = mo.lastgroup # REPLY, etc
                token = mo.group() # text
                start = mo.start()
                end = mo.end()
                if token_type in self.required_token_types:
                    tokens.append(token)
                    if self.word_to_idx is not None:
                        input_ids.append(self.word_to_idx.get(token, 0))
                    attention_mask.append(1)
                    token_type_ids.append(0)
                    for char_idx in range(start, end):
                        char_to_token_dict[char_idx] = cur_token_idx
                    cur_token_idx += 1
        return TokensEncoded(
            tokens=tokens, 
            char_to_token_dict=char_to_token_dict,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, 
        )

    def convert_ids_to_tokens(self, ids):
        assert(self.idx_to_word is not None)
        return [self.idx_to_word[int(i)] for i in ids]


        
