import sys
from pathlib import Path
from model.TDBERT import TDBERT
from model.TGSAN import TGSAN
from model.TGSAN2 import TGSAN2 


def get_model(args, state_path=None):
    Model = getattr(sys.modules[__name__], args.train_config['model_class'])
    pretrained_emb = args.model_config.get("pretrained_emb", None)
    if state_path is None :
        if pretrained_emb:
            args.pretrained_emb_path = Path("../data/word_embeddings") / pretrained_emb
        model = Model(args=args)
    else:
        model = Model(args=args)
        model.load_state(state_path=state_path)
    return model


def get_model_type(args):
    model_class = args.train_config['model_class']
    Model = getattr(sys.modules[__name__], model_class)
    return Model.MODEL_TYPE
