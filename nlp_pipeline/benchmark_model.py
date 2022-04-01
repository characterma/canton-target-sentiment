import argparse
import logging
import json
import torch
import sys
import time
import pprint
# import torch_tensorrt
from collections import OrderedDict
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from nlp_pipeline.model import get_model, get_onnx_session
from nlp_pipeline.tokenizer import get_tokenizer
from nlp_pipeline.utils import load_config, set_log_path, get_args
from nlp_pipeline.label import get_label_to_id
from nlp_pipeline.dataset import get_feature_class, get_dataset
from nlp_pipeline.dataset.utils import get_model_inputs


logger = logging.getLogger(__name__)


def benchmark_model(model_type, args):
    args = load_config(args=args)
    args.device = 0
    set_log_path(args.output_dir)
    tokenizer = get_tokenizer(args=args)
    feature_class = get_feature_class(args)

    label_to_id, label_to_id_inv = get_label_to_id(tokenizer, args)
    args.label_to_id = label_to_id
    args.label_to_id_inv = label_to_id_inv

    raw_data = json.load(open(args.data_dir / args.data_config['test'], "r"))
    for x in raw_data:
        del x['label'] 
    # data_dict = raw_data[0]
    # feature = feature_class(
    #     data_dict=data_dict, tokenizer=tokenizer, args=args, diagnosis=False
    # )

    # feature_dict = feature.feature_dict
    # # model = get_model(args=args)
    # # model.set_return_logits()

    # batch = dict()
    # for col in feature_dict:
    #     batch[col] = torch.stack([feature_dict[col]], dim=0).to(torch.int32).to(args.device)
    #     print(batch[col].device)

    # x = tuple([batch[col].squeeze(-1) for col in batch])
    
    if model_type=="original":
        model = get_model(args=args)
    elif model_type=="trace":
        model = torch.jit.load(args.model_dir / "traced_model.ts")
    elif model_type=="fp_16":
        model = torch.jit.load(args.model_dir / "trt_model_fp16.ts")
    elif model_type=="fp_32":
        model = torch.jit.load(args.model_dir / "trt_model_fp32.ts")  
    elif model_type=="onnx":
        onnx_session = get_onnx_session(args=args)
    
    if model_type != "onnx":
        model.eval()
        
    logger.info(f"***** Load {model_type} model succeeded. *****")
    dataset = get_dataset(
        raw_data=raw_data,
        dataset="test", 
        tokenizer=tokenizer, 
        args=args
    )
    
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_config["batch_size"] if args.batch_size == 0 else args.batch_size,
    )

    num_samples = len(dataset)
    t0 = time.time()

    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = dict()
            for col in batch:
                if torch.is_tensor(batch[col]):
                    inputs[col] = batch[col].to(args.device).long()
            if model_type != "onnx":
                # print(inputs)
                _ = model(**inputs)
            else:
                b = dict()
                for col in batch:
                    b[col] = batch[col].numpy()
                _ = onnx_session.run(None, input_feed=b)
            
    t1 = time.time()
    
    return (t1 - t0), num_samples



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=0)

    args = parser.parse_args()
    device = args.device
    args = load_config(args=args)
    args.device = device
    set_log_path(args.output_dir)
    time_statistics = OrderedDict()
    
    for model_type in [
        'original', 
        'trace', 
        'onnx'
    ]:
        try:
            total_time, num_samples = benchmark_model(
                model_type=model_type, 
                args=args
            )
            
            time_statistics[model_type] = dict(total_time=total_time, num_samples=num_samples, speed=num_samples / total_time)
            
        except Exception as e:
            print(e)
        
    pp = pprint.PrettyPrinter(width=41, compact=True)
    print("Results:") 
    pp.pprint(time_statistics)

    json.dump(time_statistics, open(args.config_dir / "model_speed.json", 'w'))
