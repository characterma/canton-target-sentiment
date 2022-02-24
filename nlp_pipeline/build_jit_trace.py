import argparse
import logging
import json
import torch
import sys
import numpy as np

from nlp_pipeline.model import get_model, get_onnx_session
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.utils import load_config, set_log_path, get_args
from nlp_pipeline.label import get_label_to_id
from nlp_pipeline.dataset import get_feature_class
from nlp_pipeline.dataset.utils import get_model_inputs


logger = logging.getLogger(__name__)


def build_jit_trace(args):
    args = load_config(args=args)
    set_log_path(args.output_dir)
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    data_dict = json.load(open(args.data_dir / args.data_config['test'], "r"))[0]
    feature = feature_class(
        data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
    )

    feature_dict = feature.feature_dict
    model = get_model(args=args)
    model.set_return_logits()

    batch = dict()
    for col in feature_dict:
        batch[col] = torch.stack([feature_dict[col]], dim=0).to(args.device)
        print(col, batch[col].device)
    if 'label' in batch:
        del batch['label']


    x = tuple([batch[col].squeeze(-1) for col in batch])
    model.eval()
    traced_model = torch.jit.trace(model, x)
    traced_model.save(
        str(args.model_dir / "traced_model.ts")
    )
    logger.info("***** Build traced model succeeded. *****")

    orig_output = model(**batch).cpu().detach().numpy()
    trace_output = traced_model(*batch.values()).cpu().detach().numpy()
    print("ori:", orig_output)
    print("trace:", trace_output)
    np.testing.assert_allclose(orig_output, trace_output, rtol=1e-02, atol=1e-02)
    return traced_model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../output/wbi/org_per_bert_avg_20210925_all_ext2/model")
    parser.add_argument("--device", type=str, default='cuda')

    args = parser.parse_args()
    device = args.device
    args = load_config(args=args)
    set_log_path(args.output_dir)
    args.device = device
    build_jit_trace(args=args)