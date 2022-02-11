import logging
import json
import re
import os
import random
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, BertTokenizerFast

from nlp_pipeline.preprocess import Preprocessor
from nlp_pipeline.utils import get_args, load_config
from nlp_pipeline.constants import get_constant


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
    "hfl/chinese-roberta-wwm-ext-large": BertTokenizerFast,
}


def load_vocab(tokenizer, vocab_path, args):
    logger.info("***** Loading vocab *****")
    word_to_id = json.load(open(vocab_path, "r"))
    tokenizer.update_word_id(word_to_id)
    args.vocab_size = len(word_to_id)
    logger.info("  Vocab size = %d", len(word_to_id))


def build_vocab_from_pretrained(tokenizer, args):
    emb_path = (
        Path("../data/word_embeddings") / args.model_config["pretrained_word_emb"]
    )
    words = []
    logger.info("***** Building vocab from pretrained *****")
    logger.info("  Embedding path = %s", str(emb_path))
    with open(emb_path, encoding="utf-8", errors="ignore") as f:
        for _ in f:
            break
        for line in tqdm(f):
            words.append(line.split(" ")[0])
    word_to_id = {}
    word_to_id["<PAD>"] = 0
    word_to_id["<OOV>"] = 1
    word_to_id.update(dict(zip(words, range(1, len(words) + 1))))
    logger.info("  Vocab size = %d", len(word_to_id))
    json.dump(word_to_id, open(args.model_dir / "word_to_id.json", "w"))
    tokenizer.update_word_id(word_to_id)
    args.vocab_size = len(word_to_id)


def build_vocab_from_dataset(datasets, tokenizer, args):
    logger.info("***** Building vocab from dataset *****")
    logger.info("  Datasets = %s", str(datasets))
    # logger.info("  Number of raw samples = %d", len(raw_data))
    all_words = []
    word_to_id = dict()

    for dataset in datasets:
        filename = args.data_config[dataset]
        data_path = Path(args.data_config["data_dir"]) / filename
        raw_data = json.load(open(data_path, "r"))
        for data_dict in tqdm(raw_data):
            preprocessor = Preprocessor(
                data_dict=data_dict, steps=args.prepro_config["steps"]
            )
            all_words.extend(
                tokenizer(
                    raw_text=preprocessor.data_dict["content"],
                    max_length=None,
                    padding="max_length",
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                ).tokens
            )
    word_counter = Counter(all_words)
    vocab_freq_cutoff = args.model_config["vocab_freq_cutoff"]
    words = list(set(all_words))
    words = sorted(words)
    random.shuffle(words)
    words = sorted(words, key=lambda w: word_counter[w])
    infreq_words = words[: int(vocab_freq_cutoff * len(words))]
    logger.info("  Number of infrequency words = %d", len(infreq_words))

    word_to_id["<PAD>"] = 0
    word_to_id["<OOV>"] = 1
    
    for w in words:
        if w not in infreq_words:
            word_to_id[w] = len(word_to_id)

    logger.info("  Infrequenct words = %d", len(infreq_words))
    logger.info("  Vocab size = %d", len(word_to_id))

    json.dump(word_to_id, open(args.model_dir / "word_to_id.json", "w"))
    tokenizer.update_word_id(word_to_id)
    args.vocab_size = len(word_to_id)


def get_tokenizer(args, word_to_id=None, required_token_types=None, datasets=None):
    source = args.model_config["tokenizer_source"]
    logger.info("***** Loading tokenizer *****")
    logger.info("  Tokenizer source = '%s'", source)
    if datasets is None:
        datasets = ['train']
    # logger.info("  Tokenizer name = '%s'", name)
    if source == "transformers":
        prev_model_dir = args.model_config.get("pretrained_lm_from_prev", None)
        if prev_model_dir is None:
            name = args.model_config["tokenizer_name"]
            Tokenizer = TOKENIZER_CLASS_MAP.get(name, AutoTokenizer)
            
            tokenizer_dir = args.model_dir / "tokenizer"
            if os.path.isdir(tokenizer_dir):
                tokenizer = Tokenizer.from_pretrained(
                    str(tokenizer_dir),
                    use_fast=True,
                    add_special_tokens=True
                )
            else:
                tokenizer = Tokenizer.from_pretrained(
                    name,
                    use_fast=True,
                    add_special_tokens=True
                )

                # load extra tokens
                if args.data_config.get("extra_special_tokens"):
                    try:
                        extra_special_tokens = []
                        for st in args.data_config.get("extra_special_tokens"):
                            extra_special_tokens.extend(get_constant(st))
                        extra_special_tokens = list(set(extra_special_tokens))
                        tokenizer.add_tokens(extra_special_tokens, special_tokens=True)
                        logger.info("***** Added extra special tokens *****")
                        logger.info("  Extra special tokens = '%s'", extra_special_tokens)
                    except Exception as e:
                        logger.info("***** Failed adding extra special tokens *****")
                        logger.info("  Error = '%s'", e)

                        
                tokenizer.save_pretrained(str(tokenizer_dir))
            args.tokenizer_len = len(tokenizer)
        else:
            prev_args = get_args(prev_model_dir)
            prev_args = load_config(prev_args)
            name = prev_args.model_config["tokenizer_name"]
            Tokenizer = TOKENIZER_CLASS_MAP.get(name, AutoTokenizer)
            tokenizer = Tokenizer.from_pretrained(
                name,
                use_fast=True,
                add_special_tokens=True
            )
        
        return tokenizer

    elif source in ["internal", "char_split"]:
        if source == "internal":
            tokenizer = MultiLingualTokenizer(
                word_to_id=word_to_id, required_token_types=required_token_types
            )
        else:
            tokenizer = CharacterSplitTokenizer(word_to_id=word_to_id)
        vocab_path = args.model_dir / "word_to_id.json"
        if os.path.exists(vocab_path):
            load_vocab(tokenizer=tokenizer, vocab_path=vocab_path, args=args)
        else:
            build_vocab_from_dataset(
                datasets=datasets, tokenizer=tokenizer, args=args
            )
        args.word_to_id = tokenizer.word_to_id
        return tokenizer
    else:
        raise ValueError("Unsupported tokenizer source.")


class TokensEncoded:
    # Interface class
    def __init__(
        self,
        tokens,
        char_to_token_dict,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        self._char_to_token_dict = char_to_token_dict
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.length = [len(self.tokens)]

    def char_to_token(self, char_idx):
        return self._char_to_token_dict.get(char_idx, None)

    def __len__(self):
        return len(self.tokens)


class MultiLingualTokenizer:
    def __init__(self, word_to_id=None, required_token_types=None):
        self.word_to_id = None
        self.idx_to_word = None
        if word_to_id is not None:
            self.update_word_id(word_to_id)
        self.required_token_types = (
            ["LETTERS", "CHAR", "DIGIT"]
            if required_token_types is None
            else required_token_types
        )
        self.token_spec = [
            ("REPLY", r"(引用(?:.|\n)*?發表)|(回覆樓主:)|(回覆(?:.|\n)*?帖子)"),
            (
                "URL",
                r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)"
                r"|(?:[a-zA-Z]|[0-9]|[.]|[!*\(\),])+\.[a-zA-Z]{2,}(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]"
                r"[0-9a-fA-F]))*",
            ),
            ("EMAIL", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+[a-zA-Z0-9]+"),  #
            ("ACCOUNT", r"@[^“”？@,?!。、，\n\xa0\t\r:\#\" ]+"),
            ("DIGIT", r"(?:\d+[,.]?\d*)"),
            (
                "LETTERS",
                r"Dr\."
                "|"
                "dr\."
                "|"
                "mrs?\."
                "|"
                "Mrs?\."
                "|"
                "[a-zA-Zàâçéèêëîïôûùüÿñæœ\.'-\?]*[a-zA-Zàâçéèêëîïôûùüÿñæœ]",
            ),
            ("EMOTION", r"(\[\S+?\])|(\(emoji\))"),
            ("QUESTION", r"([？?][”\"]?[\s]*)+"),
            ("PUNCTUATION", r"([!\n\r。！？?\.][”\"]?[\s]*)+"),
            ("SPACE", r"[ ]+"),
            ("CHAR", "[\u4E00-\u9FA5]"),  # Chinese character
            ("COMMA", r"[,；;，]+"),  # Any other character
            ("SPECIAL", r"."),  # Any other character
        ]

    def update_word_id(self, word_to_id):
        self.word_to_id = word_to_id
        self.idx_to_word = dict()
        for k, v in word_to_id.items():
            self.idx_to_word[v] = k

    def pad_max_length(self, values, max_length, pad_value=0):
        d = max(max_length - len(values), 0)
        values = values + [pad_value] * d
        return values

    def __call__(
        self,
        raw_text,
        raw_text2=None, 
        max_length=None,
        truncation=True,
        padding="max_lenght",
        add_special_tokens=True,
        return_offsets_mapping=False,
        return_length=False,
    ):
        token_regex = "|".join("(?P<%s>%s)" % pair for pair in self.token_spec)
        WORD_RE = re.compile(token_regex, re.VERBOSE | re.IGNORECASE)
        tokens = []
        char_to_token_dict = dict()
        input_ids = []
        attention_mask = []
        token_type_ids = []

        cur_token_idx = 0
        for mo in WORD_RE.finditer(raw_text):
            if max_length is None or len(tokens) < max_length:
                token_type = mo.lastgroup  # REPLY, etc
                token = mo.group()  # text
                start = mo.start()
                end = mo.end()
                if token_type in self.required_token_types:
                    tokens.append(token)
                    if self.word_to_id is not None:
                        input_ids.append(self.word_to_id.get(token, 1))
                    attention_mask.append(1)
                    token_type_ids.append(0)
                    for char_idx in range(start, end):
                        char_to_token_dict[char_idx] = cur_token_idx
                    cur_token_idx += 1

        if max_length is not None and padding == "max_length":
            tokens = self.pad_max_length(tokens, max_length, pad_value=0)
            input_ids = self.pad_max_length(input_ids, max_length, pad_value=0)
            attention_mask = self.pad_max_length(
                attention_mask, max_length, pad_value=0
            )
            token_type_ids = self.pad_max_length(
                token_type_ids, max_length, pad_value=0
            )

        return TokensEncoded(
            tokens=tokens,
            char_to_token_dict=char_to_token_dict,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def convert_ids_to_tokens(self, ids):
        assert self.idx_to_word is not None
        return [self.idx_to_word[int(i)] for i in ids]


class CharacterSplitTokenizer(MultiLingualTokenizer):
    def __init__(self, word_to_id=None):
        super(CharacterSplitTokenizer, self).__init__(
            word_to_id=word_to_id, required_token_types=["CHAR"]
        )
        self.token_spec = [("SPACE", r"[ ]+"), ("CHAR", r".")]
