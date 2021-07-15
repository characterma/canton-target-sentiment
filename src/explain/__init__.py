from explain.model import *


def get_explanation_model(model, args):
    """
    """
    model_class = eval(args.explain_config['model_class'])
    explanation = model_class(
        model=model, 
        args=args
    )
    return explanation
