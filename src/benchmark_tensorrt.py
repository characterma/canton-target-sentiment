import argparse
import logging
import json
import torch
import sys
import time
import torch_tensorrt
from tqdm import tqdm, trange

from model import get_model, get_onnx_session
from tokenizer import get_tokenizer
from utils import load_config, set_log_path, get_args
from label import get_label_to_id
from dataset import get_feature_class, get_dataset
from dataset.utils import get_model_inputs
from torch.utils.data import DataLoader
import pprint
from collections import OrderedDict


logger = logging.getLogger(__name__)


def benchmark_model(model_type, args):
    args = load_config(args=args)
    args.device = "cuda"
    set_log_path(args.output_dir)
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    raw_data = json.load(open(args.data_dir / args.data_config['test'], "r"))
    for x in raw_data:
        del x['label'] 
    data_dict = raw_data[0]
    feature = feature_class(
        data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
    )

    feature_dict = feature.feature_dict
    model = get_model(args=args)
    model.set_return_logits()

    batch = dict()
    for col in feature_dict:
        batch[col] = torch.stack([feature_dict[col]], dim=0).to(torch.int32).to(args.device)
        print(batch[col].device)

    x = tuple([batch[col].squeeze(-1) for col in batch])
    
    if model_type=="original":
        model = get_model(args=args)
    elif model_type=="trace":
        model = trt_ts_module = torch.jit.load(args.model_dir / "traced_model.ts")
    elif model_type=="fp_16":
        model = trt_ts_module = torch.jit.load(args.model_dir / "trt_model_fp16.ts")
    elif model_type=="fp_32":
        model = trt_ts_module = torch.jit.load(args.model_dir / "trt_model_fp32.ts")  
    
        
    logger.info(f"***** Load {model_type} model succeeded. *****")
    dataset = get_dataset(
        raw_data=raw_data,
        dataset="test", 
        tokenizer=tokenizer, 
        args=args
    )
    
    model.eval()
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_config["batch_size"],
    )

    num_samples = len(dataset)
    t0 = time.time()

    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = dict()
            for col in batch:
                if torch.is_tensor(batch[col]):
                    inputs[col] = batch[col].to(args.device).long()
            _ = model(inputs['input_ids'], inputs['attention_mask'])
            
    t1 = time.time()
    
    return (t1 - t0), num_samples



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    device = args.device
    args = parser.parse_args()
    args = load_config(args=args)
    args.device = device
    set_log_path(args.output_dir)
    time_statistics = OrderedDict()
    
    for model_type in [
        'original', 
        'trace', 
        'fp16', 
        'fp32'
    ]:
        total_time, num_samples = benchmark_model(
            model_type=model_type, 
            args=args
        )
        
        time_statistics[model_type] = dict(total_time=total_time, num_samples=num_samples)
        
    pp = pprint.PrettyPrinter(width=41, compact=True)
    print("Results:") 
    pp.pprint(time_statistics)
