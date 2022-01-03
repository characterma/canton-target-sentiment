import unittest
import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm

sys.path.append("../src/")
from dim_reduction import load_embedding
from dim_reduction import dimension_reduction
from dim_reduction import save_embedding
from model.utils import MODEL_CLASS_MAP
from tokenizer import TOKENIZER_CLASS_MAP

class TestDimReduction(unittest.TestCase):
    def test_load_local_emb(self):
        embedding_path = "../tests/test_dim_reduction_samples/sample_word_emb.txt"
        vocab_result, embedding_result = load_embedding(embedding_path)
        
        # load target files
        vocab_target_path = "../tests/test_dim_reduction_samples/sample_word_emb_vocab.txt"
        embedding_target_path = "../tests/test_dim_reduction_samples/sample_word_emb_embedding.txt"
        vocabs = []
        with open(vocab_target_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                vocabs.append(line.rstrip())
        embedding = []
        with open(embedding_target_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                split_result = line.rstrip().split(" ")
                embedding.append(split_result)
        vocab_target = np.array([vocabs])
        embedding_target = np.array(embedding, dtype=float)
        
        self.assertTrue(np.array_equal(vocab_result, vocab_target))
        self.assertTrue(np.array_equal(embedding_result, embedding_target))
        
    def test_load_pretrain_model_in_classmap(self):
        pretrain_path = 'bert-base-chinese'
        self.assertTrue(pretrain_path in MODEL_CLASS_MAP)
        self.assertTrue(pretrain_path in TOKENIZER_CLASS_MAP)
        
        vocab_result, embedding_result = load_embedding(pretrain_path)
        # load target files
        vocab_path = "../tests/test_dim_reduction_samples/sample_vocab.txt"
        embedding_target_path = "../tests/test_dim_reduction_samples/sample_bert_base_chinese_embedding_768d.txt"
        vocabs = []
        with open(vocab_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                vocabs.append(line.rstrip())
        embedding = []
        with open(embedding_target_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                split_result = line.rstrip().split(" ")
                embedding.append(split_result)
        embedding_target = np.array(embedding, dtype=np.float32)
        
        vocab_idx = [np.where(vocab_result[0] == word)[0][0] for word in vocabs]
        self.assertTrue(np.array_equal(embedding_result[vocab_idx], embedding_target))
        
    def test_load_pretrain_model_notin_classmap(self):
        pretrain_path = 'hfl/chinese-bert-wwm'
        self.assertTrue(pretrain_path not in MODEL_CLASS_MAP)
        self.assertTrue(pretrain_path not in TOKENIZER_CLASS_MAP)
         
        vocab_result, embedding_result = load_embedding(pretrain_path)
        # load target files
        vocab_path = "../tests/test_dim_reduction_samples/sample_vocab.txt"
        embedding_target_path = "../tests/test_dim_reduction_samples/sample_chinese_bert_wwm_embedding_768d.txt"
        vocabs = []
        with open(vocab_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                vocabs.append(line.rstrip())
        embedding = []
        with open(embedding_target_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                split_result = line.rstrip().split(" ")
                embedding.append(split_result)
        embedding_target = np.array(embedding, dtype=np.float32)
        
        vocab_idx = [np.where(vocab_result[0] == word)[0][0] for word in vocabs]
        self.assertTrue(np.array_equal(embedding_result[vocab_idx], embedding_target))
        
    def test_dimension_reduction(self):
        pretrain_path = 'bert-base-chinese'
        _, embedding = load_embedding(pretrain_path)
        embedding_num_vocab_target = embedding.shape[0]
        embedding_num_dimension_target = 24
        mode = 'PPA-PCA'
        remove_dim = 3
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
        
        self.assertLogs('Test shape of dimension reduction result')
        self.assertTrue(embedding_num_vocab_result == embedding_num_vocab_target)
        self.assertTrue(embedding_num_dimension_result == embedding_num_dimension_target)
        
        test_words = ['紅', '王', '你', '好', '嗎']
        for word in test_words:
            word_idx = np.where(_[0] == word)[0][0]
            word_emb = embedding[word_idx]
            cos_similarities = dot(word_emb, embedding.T) / (norm(word_emb)*norm(embedding, axis=1)) 
            cos_sorted_idx = np.argsort(cos_similarities)
            
            for i in range(2):
                similar_word_idx = cos_sorted_idx[-2-i]
                dissimilar_word_idx = cos_sorted_idx[i]
                # similar_word = _[0][similar_word_idx]
                # dissimilar_word = _[0][dissimilar_word_idx]
                
                cos_similarities_reduced = dot(reduced_tensor[word_idx], reduced_tensor.T) / (norm(reduced_tensor[word_idx])*norm(reduced_tensor, axis=1)) 
                self.assertGreater(cos_similarities_reduced[similar_word_idx], cos_similarities_reduced[dissimilar_word_idx])
            
    def test_save_embedding(self):
        pretrain_path = 'hfl/chinese-roberta-wwm-ext-large'
        vocab_result, embedding_result = load_embedding(pretrain_path)   
        embedding_num_vocab_target = 25
        embedding_num_dimension_target = 24
        
        save_path = "../tests/test_dim_reduction_samples/sample_roberta_wwm_large_embedding_{dim:n}d.txt".format(dim=embedding_num_dimension_target)
        save_embedding(
            embedding = embedding_result[:embedding_num_vocab_target, :embedding_num_dimension_target], 
            vocab = vocab_result[:,:embedding_num_vocab_target], 
            save_path = save_path
        )
        
        with open(save_path, 'rb') as f:
            for i, line in enumerate(f):
                inner_list = [val for val in line.decode("utf-8").split(' ')]
                if i == 0:
                    self.assertEqual(int(inner_list[0]), embedding_num_vocab_target)
                    self.assertEqual(int(inner_list[1]), embedding_num_dimension_target)
                    self.assertEqual(inner_list[-2], '')
                else:
                    self.assertEqual(len(inner_list), 1 + embedding_num_dimension_target)
                    self.assertTrue(type(inner_list[0]) is str)
                    self.assertTrue(type(float(inner_list[1])) is float)
                    self.assertTrue(type(float(inner_list[-2])) is float)
                    
                    
                    