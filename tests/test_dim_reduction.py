import unittest
import os
import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm

sys.path.append("../src/")
from dim_reduction import load_embedding
from dim_reduction import dimension_reduction
from model.utils import MODEL_CLASS_MAP
from tokenizer import TOKENIZER_CLASS_MAP

def load_local_vocab(vocab_path):
    vocabs = []
    with open(vocab_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            vocabs.append(line.rstrip())
    return np.array([vocabs])

def load_local_embedding(embedding_path):
    embedding = []
    with open(embedding_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            split_result = line.rstrip().split(" ")
            embedding.append(split_result)
    return np.array(embedding, dtype=np.float32)

class TestDimReduction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.pretrain_path = 'hfl/chinese-roberta-wwm-ext-large'
        cls.embedding_num_vocab_target = 21128
        cls.embedding_num_dimension_target = 64
        cls.reduction_mode = 'PPA-PCA'
        cls.save_path = f"../data/word_embeddings/roberta_wwm_large_embedding_{cls.embedding_num_dimension_target}d.txt"
        
        os.chdir("../src/")
        code = os.system(f"python dim_reduction.py \
                    --pretrain_path '{cls.pretrain_path}'\
                    --output_dim '{cls.embedding_num_dimension_target}' \
                    --save_path '{cls.save_path}' \
                    --reduction_mode '{cls.reduction_mode}'"
                    )

    def test_load_local_emb(self):
        embedding_path = "../data/word_embeddings/sample_word_emb.txt"
        vocab_result, embedding_result = load_embedding(embedding_path)
        
        # load target files
        vocab_target_path = "../data/word_embeddings/sample_word_emb_vocab.txt"
        embedding_target_path = "../data/word_embeddings/sample_word_emb_embedding.txt"
        vocab_target = load_local_vocab(vocab_target_path)
        embedding_target = load_local_embedding(embedding_target_path)
        
        self.assertTrue(np.array_equal(vocab_result, vocab_target))
        self.assertTrue(np.array_equal(embedding_result, embedding_target))
        
    def test_load_pretrain_model_in_classmap(self):
        pretrain_path = 'bert-base-chinese'
        self.assertTrue(pretrain_path in MODEL_CLASS_MAP)
        self.assertTrue(pretrain_path in TOKENIZER_CLASS_MAP)
        
        vocab_result, embedding_result = load_embedding(pretrain_path)
        # load target files
        vocab_target_path = "../data/word_embeddings/sample_vocab.txt"
        embedding_target_path = "../data/word_embeddings/sample_bert_base_chinese_embedding_768d.txt"
        
        vocab_target = load_local_vocab(vocab_target_path)
        embedding_target = load_local_embedding(embedding_target_path)
        
        vocab_idx = [np.where(vocab_result[0] == word)[0][0] for word in vocab_target[0]]
        self.assertTrue(np.array_equal(embedding_result[vocab_idx], embedding_target))
        
    def test_load_pretrain_model_notin_classmap(self):
        pretrain_path = 'hfl/chinese-bert-wwm'
        self.assertTrue(pretrain_path not in MODEL_CLASS_MAP)
        self.assertTrue(pretrain_path not in TOKENIZER_CLASS_MAP)
         
        vocab_result, embedding_result = load_embedding(pretrain_path)
        # load target files
        vocab_target_path = "../data/word_embeddings/sample_vocab.txt"
        embedding_target_path = "../data/word_embeddings/sample_chinese_bert_wwm_embedding_768d.txt"
        
        vocab_target = load_local_vocab(vocab_target_path)
        embedding_target = load_local_embedding(embedding_target_path)
        
        vocab_idx = [np.where(vocab_result[0] == word)[0][0] for word in vocab_target[0]]
        self.assertTrue(np.array_equal(embedding_result[vocab_idx], embedding_target))
        
    def test_dimension_reduction(self):
        _, reduced_tensor = load_embedding(self.save_path)
        _, embedding = load_embedding(self.pretrain_path)

        embedding_num_vocab_result = reduced_tensor.shape[0]
        embedding_num_dimension_result = reduced_tensor.shape[1]
        
        self.assertLogs('Test shape of dimension reduction result')
        self.assertTrue(embedding_num_vocab_result == self.embedding_num_vocab_target)
        self.assertTrue(embedding_num_dimension_result == self.embedding_num_dimension_target)
        
        test_words = ['紅', '王', '你', '好', '嗎']
        for word in test_words:
            word_idx = np.where(_[0] == word)[0][0]
            word_emb = embedding[word_idx]
            cos_similarities = dot(word_emb, embedding.T) / (norm(word_emb)*norm(embedding, axis=1)) 
            cos_sorted_idx = np.argsort(cos_similarities)
            
            for i in range(5):
                similar_word_idx = cos_sorted_idx[-2-i]
                dissimilar_word_idx = cos_sorted_idx[i]
                # similar_word = _[0][similar_word_idx]
                # dissimilar_word = _[0][dissimilar_word_idx]
                
                cos_similarities_reduced = dot(reduced_tensor[word_idx], reduced_tensor.T) / (norm(reduced_tensor[word_idx])*norm(reduced_tensor, axis=1)) 
                self.assertGreater(cos_similarities_reduced[similar_word_idx], cos_similarities_reduced[dissimilar_word_idx])
            
    def test_save_embedding(self):
        with open(self.save_path, 'rb') as f:
            for i, line in enumerate(f):
                inner_list = [val for val in line.decode("utf-8").split(' ')]
                if i == 0:
                    self.assertEqual(int(inner_list[0]), self.embedding_num_vocab_target)
                    self.assertEqual(int(inner_list[1]), self.embedding_num_dimension_target)
                    self.assertEqual(inner_list[-2], '')
                else:
                    self.assertEqual(len(inner_list), 1 + self.embedding_num_dimension_target)
                    self.assertTrue(type(inner_list[0]) is str)
                    self.assertTrue(type(float(inner_list[1])) is float)
                    self.assertTrue(type(float(inner_list[-2])) is float)
    @classmethod                
    def tearDownClass(cls): 
        os.system(f"rm {cls.save_path}")                  
                    