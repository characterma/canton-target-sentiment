import logging
import json
import re
import os
import unicodedata
import random
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, BertTokenizer, BertTokenizerFast
from preprocess import TextPreprocessor


logger = logging.getLogger(__name__)


TOKENIZER_CLASS_MAP = {
    "toastynews/electra-hongkongese-large-discriminator": AutoTokenizer,
    "toastynews/xlnet-hongkongese-base": AutoTokenizer,
    "xlnet-base-cased": AutoTokenizer,
    "xlm-roberta-base": AutoTokenizer,
    "xlm-roberta-large": AutoTokenizer,
    "bert-large-cased": AutoTokenizer,
    "bert-base-cased": AutoTokenizer,  # https://huggingface.co/transformers/model_doc/bert.html#bertconfig
    "bert-base-uncased": AutoTokenizer,
    "bert-base-multilingual-cased": AutoTokenizer,  # tested
    "bert-base-multilingual-uncased": AutoTokenizer,
    "bert-base-chinese": AutoTokenizer,  # testing
    "denpa92/bert-base-cantonese": BertTokenizerFast,  # missing tokenizer.
    "voidful/albert_chinese_tiny": BertTokenizerFast,  #
    "clue/albert_chinese_tiny": BertTokenizerFast,  #
    "voidful/albert_chinese_small": BertTokenizerFast,  #
    "clue/albert_chinese_small": BertTokenizerFast,
    "voidful/albert_chinese_base": BertTokenizerFast,  #
}


def load_vocab(tokenizer, vocab_path, args):
    logger.info("***** Loading vocab *****")
    word_to_idx = json.load(open(vocab_path, 'r'))
    tokenizer.update_word_idx(word_to_idx)
    args.vocab_size = len(word_to_idx)
    logger.info("  Vocab size = %d", len(word_to_idx))


def build_vocab_from_pretrained(tokenizer, args):
    emb_path = Path("../data/word_embeddings") / args.model_config['pretrained_word_emb']
    words = []
    logger.info("***** Building vocab from pretrained *****")
    logger.info("  Embedding path = %s", str(emb_path))
    with open(emb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            break
        for line in tqdm(f):
            words.append(line.split(' ')[0])
    word_to_idx = {}
    word_to_idx['<OOV>'] = 0
    word_to_idx.update(dict(zip(words, range(1, len(words) + 1))))
    logger.info("  Vocab size = %d", len(word_to_idx))
    json.dump(word_to_idx, open(args.model_dir / 'word_to_idx.json', 'w'))
    tokenizer.update_word_idx(word_to_idx)
    args.vocab_size = len(word_to_idx)


def build_vocab_from_dataset(dataset, tokenizer, args):
    filename = args.data_config[dataset]
    data_path = (
        Path(args.data_config["data_dir"]) / filename
    )
    raw_data = json.load(open(data_path, "r"))
    logger.info("***** Building vocab from dataset *****")
    logger.info("  Data path = %s", str(data_path))
    logger.info("  Number of raw samples = %d", len(raw_data))
    all_words = []
    word_to_idx = dict()

    for idx, data_dict in tqdm(enumerate(raw_data)):
        preprocessor = TextPreprocessor(
            text=data_dict['content'], 
            target_locs=data_dict['target_locs'], 
            steps=args.prepro_config['steps']
        )
        preprocessed_text = preprocessor.preprocessed_text
        all_words.extend(tokenizer(raw_text=preprocessed_text, max_length=None).tokens)
    word_counter = Counter(all_words)
    vocab_freq_cutoff = args.model_config['vocab_freq_cutoff']
    words = list(set(all_words))
    random.shuffle(words)
    words = sorted(words, key=lambda w: word_counter[w])
    infreq_words = words[:int(vocab_freq_cutoff * len(words))]
    logger.info("  Number of infrequency words = %d", len(infreq_words))

    word_to_idx['<OOV>'] = 0
    cur_idx = 1
    for w in words:
        if w not in infreq_words:
            word_to_idx[w] = cur_idx
            cur_idx += 1

    logger.info("  Infrequenct words = %d", len(infreq_words))
    logger.info("  Vocab size = %d", len(word_to_idx))

    json.dump(word_to_idx, open(args.model_dir / 'word_to_idx.json', 'w'))
    tokenizer.update_word_idx(word_to_idx)
    args.vocab_size = len(word_to_idx)


def get_tokenizer(args, word_to_idx=None, required_token_types=None):
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
        tokenizer = InternalTokenizer(word_to_idx=word_to_idx, required_token_types=required_token_types)
        vocab_path = args.model_dir / 'word_to_idx.json'
        if os.path.exists(vocab_path):
            load_vocab(tokenizer=tokenizer, vocab_path=vocab_path, args=args)
        else:
            if args.model_config.get("pretrained_emb", None) is not None:
                build_vocab_from_pretrained(tokenizer=tokenizer, args=args)
            else:
                build_vocab_from_dataset(dataset="train", tokenizer=tokenizer, args=args)
        return tokenizer
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

    def __call__(self, raw_text, max_length=None, truncation=True, add_special_tokens=True, return_offsets_mapping=False):
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


        
