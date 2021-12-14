import argparse
import logging
import os

import numpy as np
from model.utils import load_pretrained_bert, MODEL_CLASS_MAP
from pathlib import Path
from sklearn.decomposition import PCA
from tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

def load_embedding(pretrain_path: str):
    '''
        input:
        - pretrain_path: str (pretrained model name OR local path)
            Examples:
            1. pretrained model: 'hfl/chinese-roberta-wwm-ext-large'
            2. local embedding path: "../data/word_embeddings/sample_word_emb.txt"
                    text file, format:  
                    introduction line          
                    vocab1 0.2154 0.5796 0.5844 -0.5544 0.5462 ...
                    vocab2 ... 
            # TODO: ask nlp team if it it is required
            3. local model path: "../config/examples/sequence_classification/BERT_AVG/model/" (Only available for huggingface model structure)

        output:
        - numpy.array (fit output of _load_pretrained_emb)
    '''
    model_class = list(MODEL_CLASS_MAP.keys())
    if pretrain_path[-4:] == '.txt':
        # similar as _load_pretrained_emb but vocab list is required
        logger.info("***** Loading pretrained embeddings from local file *****")
        vocabs = []
        vectors = []
        with open(pretrain_path, encoding="utf-8", errors="ignore") as f:
            for _ in f:
                break
            for line in f:
                split_result = line.rstrip().split(" ")
                vocabs.append(split_result[0])
                vectors.append(split_result[1:])
        vectors = np.array(vectors, dtype=float)
        vocabs = np.array([vocabs])
        logger.info("  Embeddings size = '%s'", str(vectors.shape))
        return vocabs, vectors
    elif pretrain_path in model_class:
        logger.info("***** Loading pretrained embeddings from HuggingFace *****")
        class arg():
            def __init__(self, ppath: str):
                self.model_config = {
                    "pretrained_lm": None if os.path.exists(Path(ppath)) else ppath,
                    "pretrained_lm_from_prev": Path(ppath) if os.path.exists(Path(ppath)) else None,
                    "tokenizer_source": "transformers",
                    "tokenizer_name": pretrain_path
                }
        args = arg(pretrain_path)
        model = load_pretrained_bert(args)
        param_dict = dict(model.named_parameters())
        # embedding key name in model (huggingface: embeddings.word_embeddings.weight)
        emb_key = 'embeddings.word_embeddings.weight'
        vectors = param_dict[emb_key].cpu().detach().numpy()
        tokenizer = get_tokenizer(args)
        vocab_dict = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
        vocabs = np.array([list(vocab_dict.keys())])
        return vocabs, vectors
    else:
        logging.error('pretrain path does not exist in local directory or huggingface selected models.')
        return

def post_processing_algorithm(embedding: np.array, output_dim: int, remove_dim: int, seed: int):
    logger.info("***** PPA : Removing Projections on Top "+str(remove_dim)+" Components *****")
    # substract Mean Embedding
    embedding = embedding - np.mean(embedding)

    # Compute PCA components
    pca =  PCA(n_components = output_dim, random_state = seed)
    X_fit = pca.fit_transform(embedding)
    U1 = pca.components_
    z = []

    # Remove Top-D Components 
    for i, x in enumerate(embedding):
        for u in U1[0:remove_dim]:        
            x = x - np.dot(u.transpose(),x) * u 
        z.append(x)
    z = np.asarray(z)
    return z

def principal_components_analysis(embedding: np.array, output_dim: int, seed: int):
    logger.info("***** PCA: Dimension Reduction *****")
    pca =  PCA(n_components = output_dim, random_state = seed)
    embedding = embedding - np.mean(embedding)
    z = pca.fit_transform(embedding)
    return z

def dimension_reduction(embedding: np.array, output_dim: int, mode: str, remove_dim: int, seed: int):
    '''
        input:
        - embedding: np.array
        - output_dim: int
        - mode: str
        - remove_dim: int
        
        output:
        - numpy.array
    '''
    if embedding.ndim != 2:
        logging.error('Embedding tensor must be 2 dimensions')
        return
    if embedding.shape[1] < 2 * output_dim:
        logging.error('Two times of Reduced dimension must be equal or less than embedding dimensions')
        return
    if embedding.shape[1] < remove_dim:
        logging.error('Embedding dimension must be equal or more than '+remove_dim)
        return
    if embedding.shape[0] < 2 * output_dim:
        logging.error('Two times of reduced dimension must be equal or less than no. of vocab')
        return

    seed = 42

    if mode.split('-')[0] == 'PPA':
        embedding = post_processing_algorithm(embedding, output_dim*2, remove_dim, seed)
    embedding = principal_components_analysis(embedding, output_dim, seed)
    if mode.split('-')[-1] == 'PPA':   
        embedding = post_processing_algorithm(embedding, output_dim, remove_dim, seed)
    return embedding

def save_embedding(embedding: np.array, vocab: np.array, save_path: str):
    '''
        input:
        - embedding: np.array
        - vocab: np.array
        - save_path: str (local path)
            - local path: "../data/word_embeddings/sample_word_emb.txt"
    '''
    logger.info("***** Save embedding file *****")
    result = np.concatenate((vocab.T, embedding), axis=1)
    headline = np.full_like(result[0], '', dtype='<U80')
    headline[0] = embedding.shape[0]
    headline[1] = embedding.shape[1]
    result = np.vstack([headline, result])
    np.savetxt(save_path, result, delimiter=' ', fmt='%s')

def run():
    ### Load module
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_path", type=str, required=True)
    parser.add_argument("--output_dim", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--reduction_mode", 
        type=str, 
        default='PPA-PCA-PPA', 
        choices=['PPA-PCA', 'PCA-PPA', 'PCA', 'PPA-PCA-PPA']
    )
    parser.add_argument("--num_remove_principal", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vocab, embedding_tensor = load_embedding(pretrain_path=args.pretrain_path)
    
    reduced_tensor = dimension_reduction(
        embedding = embedding_tensor, 
        output_dim = args.output_dim, 
        mode = args.reduction_mode,
        remove_dim = args.num_remove_principal,
        seed = args.seed
    )
 
    save_embedding(embedding = reduced_tensor, vocab=vocab, save_path = args.save_path)

if __name__ == "__main__":
    run()