import argparse
import logging
import json
import torch

from model import get_model, get_onnx_session
from tokenizer import get_tokenizer
from utils import load_config, set_log_path
from label import get_label_to_id
from dataset import get_feature_class
from dataset.utils import get_model_inputs


logger = logging.getLogger(__name__)

    
def build_onnx(args):  
    logger.info("***** Build onnx started. *****")
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    data_dict = json.load(open(args.data_dir / args.data_config['test'], "r"))[0]
    if 'label' in data_dict:
        del data_dict['label']

    feature = feature_class(
        data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
    )

    feature_dict = feature.feature_dict
    model = get_model(args=args)
    batch = dict()
    for col in feature_dict:
        batch[col] = torch.stack([feature_dict[col]], dim=0).to(args.device)

    output = model(**batch)
    x = tuple([batch[col].squeeze(-1) for col in batch])
    model_inputs = batch.keys()

    logger.info("***** Exporting onnx model. *****")
    dynamic_axes = dict()

    for col in feature_dict:
        dynamic_axes[col] = [0]

    torch.onnx.export(
        model, 
        args=x, 
        dynamic_axes=dynamic_axes,
        f=args.model_dir / "model.onnx", 
        do_constant_folding=True, 
        opset_version=12, 
        input_names=list(model_inputs), 
        output_names=['outputs']
    )

    logger.info("***** Testing onnx model. *****")
    session = get_onnx_session(args=args)
    batch = dict()
    for col in feature_dict:
        batch[col] = feature_dict[col].unsqueeze(0).numpy()
    if 'label' in batch:
        del batch['label']

    output = session.run(None, input_feed=batch)
    logger.info("***** Build onnx succeeded. *****")
    logger.info("  Output = %s", output)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../output/wbi/org_per_bert_avg_20210925_all_ext2/model")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = args.device
    args = load_config(args=args)
    set_log_path(args.output_dir)
    args.device = device
    build_onnx(args=args)



