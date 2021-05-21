import sys
from model.TDBERT import TDBERT
from model.TGSAN import TGSAN
from model.TGSAN2 import TGSAN2 


def get_model(args, state_path=None):
    model_class = args.train_config['model_class']
    Model = getattr(sys.modules[__name__], model_class)

    # load pretrained emb
    
    model = Model(args=args)
    if state_path is not None:
        model.load_state(state_path=state_path)
    return model


def get_model_type(args):
    model_class = args.train_config['model_class']
    Model = getattr(sys.modules[__name__], model_class)
    return Model.MODEL_TYPE
