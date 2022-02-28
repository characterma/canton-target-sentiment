import argparse
import logging

from utils import load_config, set_log_path

from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOnnxModel
from onnxruntime.quantization import quantize_dynamic, QuantType


logger = logging.getLogger(__name__)

    
def optimize_onnx_graph(args):  
    logger.info("***** Optimize onnx graph started. *****")
    model_path = str(args.model_dir / "model.onnx")

    optimization_options = FusionOptions('bert')
    optimization_options.enable_relu_approximation = True  # additional optimization
    optimization_options.enable_embed_layer_norm = False # reduce embedding size
    optimized_model: BertOnnxModel = optimizer.optimize_model(
        input=model_path,
        model_type="bert",
        use_gpu=False,
        opt_level=1,
        num_heads=0,  # automatic detection
        hidden_size=0,  # automatic detection
        optimization_options=optimization_options,
    )
    optimized_model.save_model_to_file(model_path)

    logger.info("***** Optimize onnx graph succeeded. *****")


def quantize_onnx(args):
    logger.info("***** Start optimizing ONNX graph with Quantization *****")
    model_path = str(args.model_dir / "model.onnx")
    quantize_dynamic(model_path,
                    model_path,
                    op_types_to_quantize=['MatMul', 'Attention'],
                    weight_type=QuantType.QInt8,
                    per_channel=True,
                    reduce_range=True,
                    nodes_to_exclude=[],
                    extra_options={'WeightSymmetric': False, 'MatMulConstBOnly': True})
    logger.info("***** Finished optimizing ONNX graph with quantization *****")


def optimize(args):
    optimize_onnx_graph(args)
    quantize_onnx(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=str, default="../output/apple_care/indomain_model_4/model/")
    args = parser.parse_args()
    args = load_config(args=args)
    set_log_path(args.output_dir)
    optimize(args=args)
