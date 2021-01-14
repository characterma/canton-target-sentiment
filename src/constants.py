SENTI_ID_MAP = {
    # "unknown": -1,
    "neutral": 0,
    "negative": 1,
    "positive": 2,
}

SENTI_ID_MAP_INV = {}
for k, v in SENTI_ID_MAP.items():
    SENTI_ID_MAP_INV[v] = k

MODEL_EMB_TYPE = {
    "TGSAN": "WORD",
    "TDLSTM": "WORD",
    "ATAE_LSTM": "WORD",
    "IAN": "WORD",
    "MEMNET": "WORD",
    "RAM": "WORD",
    "TNET_LF": "WORD",
    "TDBERT": "BERT",
}


class SPEC_TOKEN:
    TARGET = "[TGT]"
