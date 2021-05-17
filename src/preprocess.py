# coding=utf-8
import re
import numpy as np
from hanziconv import HanziConv


class TextPreprocessor:
    def __init__(self, text, target_locs, steps):
        self.preprocessed_text = str(text) 
        self.preprocessed_target_locs = target_locs.copy()
        for s in steps:
            getattr(self, s)()

    def simplified_chinese(self):
        self.preprocessed_text = HanziConv.toSimplified(self.preprocessed_text)

    def lower_case(self):
        self.preprocessed_text = self.preprocessed_text.lower()

    def convert_java_index(self):
        index_map = dict()
        indent = 0
        for i in range(len(self.preprocessed_text)):
            for j in range(i, i + indent + 1):
                if j not in index_map:
                    index_map[j] = i
            if len(self.preprocessed_text[i].encode('utf-8')) == 4:
                indent += 1
        for t in self.preprocessed_target_locs:
            # print(t[0])
            t[0] = index_map[t[0]]
            # print(t[0])
            t[1] = index_map[t[1]]

    def extract_post_context(self, n_prev=0, n_next=0):
        """
        """
        def find_idices_after_split(start_idx, end_idx, sents):
            text_len = end_idx - start_idx
            cur_start_idx = 0
            cur_end_idx = 0
            for i, s in enumerate(sents):
                cur_end_idx += len(s) + 1
                if cur_start_idx <= start_idx < cur_end_idx:
                    if start_idx - cur_start_idx + text_len <= len(s):
                        return i, start_idx - cur_start_idx, start_idx - cur_start_idx + text_len
                    else:
                        return None, None, None
                        
                cur_start_idx += len(s) + 1
            return None, None, None
            
        def neighbor_sentences(sentence_id, sents, n_prev=0, n_next=0):
            prev = []
            next = []
            sents_with_id = list(zip(range(len(sents)), sents))
            for s in sents_with_id[:sentence_id][-1::-1]:
                if s[1].strip()!="":
                    prev.append(s[0])

            if sentence_id!=len(sents)-1:
                for s in sents_with_id[sentence_id+1:]:
                    if s[1].strip()!="":
                        next.append(s[0])
            
            prev = prev[:n_prev][-1::-1]
            next = next[:n_next]
            return prev, next

        sents = re.split("\?|!|。|？|！|\n|\r", self.preprocessed_text)
        reconst_sent_ids = []
        reconst_target_indices = []
        cur_len = 0

        for start_idx, end_idx in self.preprocessed_target_locs:
            
            sentence_id, start_idx, end_idx = find_idices_after_split(
                start_idx,
                end_idx,
                sents
            )
            
            prev_sents, next_sents = neighbor_sentences(sentence_id, sents, n_prev=n_prev, n_next=n_next)

            cur_start_idx = start_idx + cur_len
            cur_end_idx = end_idx + cur_len
            for idx in prev_sents:
                if idx not in reconst_sent_ids and idx!=sentence_id:
                    reconst_sent_ids.append(idx)
                    cur_start_idx += len(sents[idx])
                    cur_end_idx += len(sents[idx])
                    cur_len += len(sents[idx])
                    
            reconst_sent_ids.append(sentence_id)
            cur_len += len(sents[sentence_id])
            
            for idx in next_sents:
                if idx not in reconst_sent_ids and idx!=sentence_id:
                    reconst_sent_ids.append(idx)
                    cur_len += len(sents[idx])
            reconst_target_indices.append([cur_start_idx, cur_end_idx])
            
        reconst_content = "".join([sents[i] for i in reconst_sent_ids])

        self.preprocessed_text = reconst_content
        self.preprocessed_target_locs = reconst_target_indices
            