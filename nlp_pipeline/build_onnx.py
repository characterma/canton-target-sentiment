import argparse
import logging
import json
import torch
import numpy as np

from nlp_pipeline.model import get_model, get_onnx_session
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.utils import load_config, set_log_path
from nlp_pipeline.label import get_label_to_id
from nlp_pipeline.dataset import get_feature_class
from nlp_pipeline.dataset.utils import get_model_inputs


logger = logging.getLogger(__name__)


TASK_TO_OUTPUT_SHAPE = {
    "sequence_classification": {0: 'batch_size'}, 
    "target_classification": {0: 'batch_size', 1: 'max_seq_len',}, 
}
    
def build_onnx(args):  
    logger.info("***** Build onnx started. *****")
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    # Load data
    data_dict = json.load(open(args.data_dir / args.data_config['test'], "r"))[0]
    if 'label' in data_dict:
        del data_dict['label']
    feature = feature_class(
        data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
    )
    feature_dict = feature.feature_dict

    # Load model
    model = get_model(args=args)
    model.set_return_logits()

    # Get model input fields
    model_inputs = []
    for i in get_model_inputs(args=args):
        if i in feature_dict.keys():
            model_inputs.append(i)

    logger.info("***** Exporting onnx model. *****")

    # Dynamic axes
    dynamic_axes = dict()
    for col in model_inputs:
        dynamic_axes[col] = {0: 'batch_size', 1: 'max_seq_len',}
    dynamic_axes['outputs'] = TASK_TO_OUTPUT_SHAPE[args.task]
    
    # Make input for onnx export
    batch = dict()
    for col in model_inputs:
        batch[col] = torch.stack([feature_dict[col]], dim=0).to(args.device)
    x = tuple([batch[col].squeeze(-1) for col in batch])

    # Export & load onnx
    model.eval()
    with torch.no_grad():
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

    # Test onnx output vs original model output
    for bz in [1, 2, 8, 16]:

        for l in [0.5, 1, 2]:

            batch_orig = dict()
            batch_onnx = dict()
            # print(bz, l)
            for col in model_inputs:
                org_tensor = feature_dict[col].unsqueeze(0)
                repeat_size = [1] * len(org_tensor.size())
                repeat_size[0] = bz 

                if  l >= 1:
                    assert(type(l) is int)
                    repeat_size[1] = l 

                    batch_orig[col] = feature_dict[col].unsqueeze(0).repeat(*repeat_size).to(args.device)
                    batch_onnx[col] = feature_dict[col].unsqueeze(0).repeat(*repeat_size).numpy()

                else:
                    seq_len = org_tensor.size()[1]
                    batch_orig[col] = feature_dict[col].unsqueeze(0).repeat(*repeat_size)[:, :int(seq_len * l)].to(args.device)
                    batch_onnx[col] = feature_dict[col].unsqueeze(0).repeat(*repeat_size)[:, :int(seq_len * l)].numpy()

                # print(col, batch_onnx[col].shape)

            orig_output = model(**batch_orig).cpu().detach().numpy()
            onnx_output = session.run(None, input_feed=batch_onnx)
            print("ori:", orig_output)
            print("onx:", onnx_output)
            onnx_output = np.array(onnx_output[0])
            np.testing.assert_allclose(orig_output, onnx_output, rtol=1e-02, atol=1e-02)

    logger.info("***** Build onnx succeeded. *****")


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



