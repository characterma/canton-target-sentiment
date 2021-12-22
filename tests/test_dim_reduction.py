import unittest
import sys
import numpy as np
from numpy import loadtxt
from transformers import AutoTokenizer, AutoModel

sys.path.append("../src/")
from dim_reduction import load_embedding
from dim_reduction import dimension_reduction
from dim_reduction import save_embedding
from model.utils import MODEL_CLASS_MAP
from tokenizer import TOKENIZER_CLASS_MAP

class TestDimReduction(unittest.TestCase):
    def test_load_local_emb(self):
        pretrain_path = "../data/word_embeddings/sample_word_emb.txt"
        vocab_result, embedding_result = load_embedding(pretrain_path)
        
        vocabs = []
        embedding = []
        with open(pretrain_path, encoding="utf-8", errors="ignore") as f:
            for _ in f:
                break
            for line in f:
                split_result = line.rstrip().split(" ")
                vocabs.append(split_result[0])
                embedding.append(split_result[1:])
        embedding_target = np.array(embedding, dtype=float)
        vocab_target = np.array([vocabs])
        
        self.assertTrue(np.array_equal(vocab_result, vocab_target))
        self.assertTrue(np.array_equal(embedding_result, embedding_target))
        
    def test_load_pretrain_model_in_classmap(self):
        pretrain_path = 'hfl/chinese-roberta-wwm-ext-large'
        self.assertTrue(pretrain_path in MODEL_CLASS_MAP)
        self.assertTrue(pretrain_path in TOKENIZER_CLASS_MAP)
        
        vocab_result, embedding_result = load_embedding(pretrain_path)
        tokenizer = TOKENIZER_CLASS_MAP[pretrain_path].from_pretrained(
            pretrain_path,
            use_fast=True,
            add_special_tokens=True
        )
        vocab_dict = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
        vocab_target = np.array([list(vocab_dict.keys())])
        model = MODEL_CLASS_MAP[pretrain_path].from_pretrained(pretrain_path)
        model.resize_token_embeddings(len(tokenizer))
        embedding_target = dict(model.named_parameters())['embeddings.word_embeddings.weight'].cpu().detach().numpy()
        
        self.assertTrue(np.array_equal(vocab_result, vocab_target))
        self.assertTrue(np.array_equal(embedding_result, embedding_target))
        
    def test_load_pretrain_model_notin_classmap(self):
        pretrain_path = 'hfl/chinese-bert-wwm'
        self.assertTrue(pretrain_path not in MODEL_CLASS_MAP)
        self.assertTrue(pretrain_path not in TOKENIZER_CLASS_MAP)
         
        vocab_result, embedding_result = load_embedding(pretrain_path)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrain_path,
            use_fast=True,
            add_special_tokens=True
        )
        vocab_dict = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
        vocab_target = np.array([list(vocab_dict.keys())])
        model = AutoModel.from_pretrained(pretrain_path)
        model.resize_token_embeddings(len(tokenizer))
        embedding_target = dict(model.named_parameters())['embeddings.word_embeddings.weight'].cpu().detach().numpy()
        
        self.assertTrue(np.array_equal(vocab_result, vocab_target))
        self.assertTrue(np.array_equal(embedding_result, embedding_target))
        
    def test_dimension_reduction(self):
        pretrain_path = 'hfl/chinese-roberta-wwm-ext-large'
        _, embedding = load_embedding(pretrain_path)
        embedding_num_vocab_target = embedding.shape[0]
        embedding_num_dimension_target = 16
        mode = 'PPA-PCA'
        remove_dim = 7
        seed = 42
        
        reduced_tensor = dimension_reduction(
            embedding = embedding,
            output_dim = embedding_num_dimension_target,
            mode = mode,
            remove_dim = remove_dim,
            seed = seed
        )
        embedding_num_vocab_result = reduced_tensor.shape[0]
        embedding_num_dimension_result = reduced_tensor.shape[1]
        
        self.assertTrue(embedding_num_vocab_result == embedding_num_vocab_target)
        self.assertTrue(embedding_num_dimension_result == embedding_num_dimension_target)
        
    def test_save_embedding(self):
        pretrain_path = 'hfl/chinese-roberta-wwm-ext-large'
        vocab_result, embedding_result = load_embedding(pretrain_path)   
        embedding_num_vocab_target = embedding_result.shape[0]
        embedding_num_dimension_target = embedding_result.shape[1]
        
        save_path = "../data/word_embeddings/roberta_{dim:n}D.txt".format(dim=embedding_num_dimension_target)
        save_embedding(embedding = embedding_result, vocab = vocab_result, save_path = save_path)
        
        with open(save_path, 'rb') as f:
            for i, line in enumerate(f):
                inner_list = [val for val in line.decode("utf-8").split(' ')]
                if i == 0:
                    self.assertEqual(int(inner_list[0]), embedding_num_vocab_target)
                    self.assertEqual(int(inner_list[1]), embedding_num_dimension_target)
                    self.assertEqual(inner_list[2], '')
                else:
                    self.assertEqual(len(inner_list), 1 + embedding_num_dimension_target)
                    self.assertTrue(type(inner_list[0]) is str)
                    self.assertTrue(type(float(inner_list[1])) is float)
                    
                    
                    