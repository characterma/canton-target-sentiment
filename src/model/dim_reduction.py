import os
import torch
from model.utils import load_pretrained_bert
from pathlib import Path

def load_embedding(pretrain_path: str):
    '''
        input:
        - pretrain_path: str (pretrained model name OR local path)
        - pretrained model: 'hfl/roberta-wwm-ext'
        - local path: '../config/src/glove_300t.txt'
                text file, format:            
                vocab1 0.2154 0.5796 0.5844 -0.5544 0.5462 ...
                vocab2 ...     
        output:
        - torch.tensor
    '''
    class arg():
        def __init__(self, ppath: str):
            self.model_config = {
                "pretrained_lm": None if os.path.exists(Path(ppath)) else ppath,
                "pretrained_lm_from_prev": Path(ppath) if os.path.exists(Path(ppath)) else None
            }
    args = arg(pretrain_path)
    model = load_pretrained_bert(args)
    source_param_dict = dict(model.named_parameters())
    # embedding key name in model
    # text cnn: emb
    # huggingface: embeddings.word_embeddings.weight
    emb_key = 'emb' if 'emb' in source_param_dict.keys() else 'embeddings.word_embeddings.weight'
    return source_param_dict[emb_key]

def dimension_reduction(embedding: torch.tensor, output_dim: int):
    return

def save_embedding(embeddingL torch.tensor, save_path: str):
    

def run():
    ### Load module
    '''
    input:
    - pretrain_path: str (pretrained model name OR local path)
    - pretrained model: 'hfl/roberta-wwm-ext'
    - local path: '../config/src/glove_300t.txt'
            text file, format:
        
        vocab1 0.2154 0.5796 0.5844 -0.5544 0.5462 ...
        vocab2 ...
    
    output:
    - torch.tensor
    '''
    embedding_tensor = load_embedding(pretrain_path="")
    
    ### dimension reduction
    '''
    input:
    - embedding: torch.tensor
    - output_dim: int
    
    output:
    - torch.tensor
    '''
    reduced_tensor = dimension_reduction(embedding = embedding_tensor, output_dim = 64)
    
    ### save module
    '''
    input:
    - embedding: torch.tensor
    - save_path: str (local path)
        - local path: '../config/src/glove_300t.txt'
    
    '''
    save_embedding(embedding = reduced_tensor, save_path = "")

if __name__ == "__main__":
    run()