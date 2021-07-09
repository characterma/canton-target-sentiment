# from model import get_model
# from dataset import get_feature_class
# from utils import get_args, load_config
# from label import get_label_to_id
# from tokenizer import get_tokenizer
from explain.model import CaptumExplanation


def get_explanation_model(model, args):
    """

    """
    return CaptumExplanation(
        model=model, 
        args=args,
    )
    # args = load_config(args)

    # model = get_model(args)
    # model = model.to(args.device)

    # feature_class = get_feature_class(args)
    # tokenizer = get_tokenizer(args)

    # label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    # args.label_to_id = label_to_id
    # args.label_to_id_inv = label_to_id_inv