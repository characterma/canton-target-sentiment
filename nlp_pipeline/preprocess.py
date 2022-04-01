# coding=utf-8
import re
import copy
import pickle as pkl
from copy import deepcopy
from opencc import OpenCC

from nlp_pipeline.constants.sina_emojis import sina_emojis


FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x21, 0x7F))
FULL2HALF[0x3000] = 0x20


class Preprocessor:
    def __init__(
        self, 
        data_dict, 
        steps, 
        args = None
    ):
        self.cc = OpenCC("t2s")
        self.data_dict = copy.deepcopy(data_dict)
        self.data_dict["raw"] = copy.deepcopy(data_dict)
        self.args = args

        for s in steps:
            getattr(self, s)()

    def rm_url(self):
        pattern = r'http\S+'
        old_target_locs = self.data_dict["target_locs"]
        new_target_locs = self.data_dict["target_locs"]
        content = self.data_dict["content"]
        for match in re.finditer(pattern, content):
            if match:
                new_target_locs = []
                start_idx = match.start()
                end_idx = match.end() 
                length = end_idx - start_idx
                for target in old_target_locs:
                    #  Case I: intersect
                    if ((target[0] >= start_idx and target[0] < end_idx) or (target[1] > start_idx and target[1] <= end_idx)):
                        pass 
                    #  Case II: right
                    elif target[0] >= end_idx:
                        new_target_locs.append(
                            [target[0] - length, target[1] - length]
                        )
                    #  Case III: left
                    else:
                        new_target_locs.append(target)
                old_target_locs = new_target_locs
                content = content[:start_idx] + content[end_idx:]

        self.data_dict["target_locs"] = new_target_locs
        self.data_dict["content"] = content

    def rm_non_chinese_char(self):
        filtrate = re.compile(u"[^\u4E00-\u9FA5]")
        self.data_dict["content"] = filtrate.sub(r"", self.data_dict["content"])

    def rm_emojis(self):
        updated_idx = []
        to_remove = sorted(sina_emojis, key=lambda x: len(x), reverse=True)  
        to_remove = [r.replace('[', '\[').replace(']', '\]') for r in to_remove]
        if 'target_locs' in self.data_dict:
            target_locs = deepcopy(self.data_dict['target_locs'])
        pattern = "|".join(to_remove)
        for match in re.finditer(pattern, self.data_dict['content']):
            text = self.data_dict['content']
            e = match.end()
            s = match.start()
            if 'target_locs' in self.data_dict:
                for t0, t1 in zip(self.data_dict['target_locs'], target_locs):
                    if t0[0] >= e:
                        t1[0] -= (e - s)
                        t1[1] -= (e - s)
                    elif t0[1] <= s:
                        pass
            self.data_dict['content'] = self.data_dict['content'].replace(match.group(), "")     
        if 'target_locs' in self.data_dict:
            self.data_dict['target_locs'] = target_locs

    def normalize_target(self):
        target_locs = sorted(self.data_dict['target_locs'], key=lambda x: x[0])
        content = self.data_dict['content']
        replacement = "[unused1]"
        
        shift = 0
        for t in target_locs:
            t[0] += shift
            t[1] += shift
            content = content[:t[0]] + replacement + content[t[1]:]
            shift += len(replacement) - (t[1] - t[0])
            t[1] = t[0] + len(replacement)
            
        self.data_dict['target_locs'] = target_locs
        self.data_dict['content'] = content

    def target_sentence_truncate(self):
        assert self.args is not None
        max_length = self.args.model_config['max_length']

        target_locs = sorted(self.data_dict['target_locs'], key=lambda x: x[0])
        if target_locs[0][1] >= max_length:
            offset = target_locs[0][0] - int((max_length - 2) / 2)
            content = self.data_dict['content'][offset:max_length+offset]
            for t in target_locs:
                t[0] -= offset
                t[1] -= offset
            self.data_dict['target_locs'] = target_locs
            self.data_dict['content'] = content

    def enclose_target(self):
        content = self.data_dict['content']
        left_token = "[E]"
        right_token = "[/E]"

        target_locs = []
        shift = 0
        for t in sorted(self.data_dict['target_locs'], key=lambda x: x[0]):
            t[0] += shift
            t[1] += shift
            content = content[:t[0]] + left_token + content[t[0]:t[1]] + right_token + content[t[1]:]
            shift += len(left_token) + len(right_token)
            target_locs.append(
                [t[0], t[0] + len(left_token)]
            )
            
        self.data_dict['content'] = content
        self.data_dict['target_locs'] = target_locs

    def mask_other_targets(self):
        content = self.data_dict['content']
        mask_token = "[unused2]"

        for i, t1 in enumerate(self.data_dict.get('other_target_locs', [])):
            to_mask = True
            for t0 in self.data_dict['target_locs']:
                x0 = range(t0[0], t0[1])
                x1 = range(t1[0], t1[1])
                if len(set(x0).intersection(x1)) > 0:
                    to_mask = False
                    
            if to_mask:
                content = content[:t1[0]] + mask_token + content[t1[1]:]
                shift = len(mask_token) - (t1[1] - t1[0])
                for t0 in self.data_dict['target_locs']:
                    if t0[0] > t1[0]:
                        t0[0] += shift
                        t0[1] += shift

                for t2 in self.data_dict['other_target_locs'][i + 1:]:
                    if t2[0] > t1[0]:
                        t2[0] += shift
                        t2[1] += shift   

        self.data_dict['content'] = content

    def full_to_half(self):
        self.data_dict["content"] = self.data_dict["content"].translate(FULL2HALF)

    def utf8_replace(self):
        self.data_dict["content"] = (
            self.data_dict["content"].encode("utf-8", errors="replace").decode("utf-8")
        )

    def add_categorical_features(self):
        self.data_dict["content"] = (
            self.data_dict["content"].encode("utf-8", errors="replace").decode("utf-8")
        )

    def simplified_chinese(self):
        self.data_dict["content"] = self.cc.convert(self.data_dict["content"])

    def lower_case(self):
        self.data_dict["content"] = self.data_dict["content"].lower()

    def concat_headline_content_with_sep(self):
        sep_token = " [SEP]"
        self.data_dict["content"] = sep_token.join(
            [self.data_dict["headline"], self.data_dict["content"]]
        )
        add_len = len(self.data_dict["headline"]) + len(sep_token)
        self.data_dict["target_locs"] = self.data_dict['target_locs_hl'] + [[x[0] + add_len, x[1] + add_len] for x in self.data_dict['target_locs_ct']]

    def concat_pub_code(self):
        sep_token = " [SEP]"
        self.data_dict["content"] = sep_token.join(
            [self.data_dict["pub_code"], self.data_dict["content"]]  #  try both directions
        )
        add_len = len(self.data_dict["pub_code"]) + len(sep_token)
        self.data_dict["target_locs"] = [[x[0] + add_len, x[1] + add_len] for x in self.data_dict['target_locs']]

    def concat_entity_name(self):
        sep_token = " [SEP]"
        self.data_dict["content"] = sep_token.join(
            [self.data_dict["entity"], self.data_dict["content"]]  #  try both directions
        )
        add_len = len(self.data_dict["entity"]) + len(sep_token)
        self.data_dict["target_locs"] = [[x[0] + add_len, x[1] + add_len] for x in self.data_dict['target_locs']]

    def convert_java_index(self):
        assert "content" in self.data_dict
        assert "target_locs" in self.data_dict

        index_map = dict()
        indent = 0
        for i in range(len(self.data_dict["content"]) + 1):
            for j in range(i, i + indent + 1):
                if j not in index_map:
                    index_map[j] = i
            if i < len(self.data_dict["content"]):
                if len(self.data_dict["content"][i].encode("utf-8")) == 4:
                    indent += 1

        for t in self.data_dict["target_locs"]:
            t[0] = index_map.get(t[0], t[0])
            t[1] = index_map.get(t[1], t[1])

    def extract_post_context_1(self):
        assert "content" in self.data_dict
        assert "target_locs" in self.data_dict
        return self._extract_post_context(n_prev=0, n_next=0)

    def extract_post_context_2(self):
        assert "content" in self.data_dict
        assert "target_locs" in self.data_dict
        return self._extract_post_context(n_prev=1, n_next=1)

    def _extract_post_context(self, n_prev=0, n_next=0):

        def find_idices_after_split(start_idx, end_idx, sents):
            text_len = end_idx - start_idx
            cur_start_idx = 0
            cur_end_idx = 0
            for i, s in enumerate(sents):
                cur_end_idx += len(s)
                if cur_start_idx <= start_idx and end_idx <= cur_end_idx:
                    return (
                        i,
                        start_idx - cur_start_idx,
                        start_idx - cur_start_idx + text_len,
                    )
                cur_start_idx = cur_end_idx
            return None, None, None

        def neighbor_sentences(sentence_id, sents, n_prev=0, n_next=0, split_text=None):
            prevs = []
            nexts = []
            if split_text is None:
                split_text = []
            sents_with_id = list(zip(range(len(sents)), sents))

            next_cnt = 0
            prev_cnt = 0
            for idx, text in sents_with_id[:sentence_id][-1::-1]:
                if prev_cnt >= n_prev:
                    break
                if text.strip() == "":
                    continue
                if text not in split_text:
                    prevs.append(idx)
                    if idx + 1 < len(sents):
                        prevs.append(idx + 1)
                    prev_cnt += 1

            if sentence_id != len(sents) - 1:
                for idx, text in sents_with_id[sentence_id + 1 :]:
                    if next_cnt >= n_next:
                        break
                    if text.strip() == "":
                        continue
                    if text not in split_text:
                        nexts.append(idx)
                        if idx + 1 < len(sents):
                            nexts.append(idx + 1)
                        next_cnt += 1

            return prevs, nexts

        def get_base_content(keep_before, sents, split_text):
            base_sids = []
            tmp_len = 0
            for sid, st in enumerate(sents):
                st = st.strip()
                tmp_len += len(st)
                if st != "" and tmp_len <= keep_before:
                    base_sids.append(sid)
            return base_sids

        def get_keep_before_idx(text, keep_before_text):
            keep_before_idx = 0
            for s in keep_before_text:
                for m in re.finditer(s, text):
                    if m and m.end() > keep_before_idx:
                        keep_before_idx = m.end() 
            return keep_before_idx

        def reconstruct_target_locs(reconst_sents, sents, tid_to_sid, tid_to_locs):
            sid_to_tid = {}
            for sid in reconst_sents:
                sid_to_tid[sid] = []
            for tid, sid in tid_to_sid.items():
                sid_to_tid[sid].append(tid)

            target_locs = []
            tids = []
            cur_len = 0
            for sid in reconst_sents:
                for tid in sid_to_tid[sid]:
                    start_idx, end_idx = tid_to_locs[tid]
                    target_locs.append(
                        [start_idx + cur_len, end_idx + cur_len]
                    )
                    tids.append(tid)
                cur_len += len(sents[sid])
            return target_locs, tids

        def validate_target_locs(reconst_content, reconst_target_locs, tids, tid_to_text):
            reconst_target_texts = [reconst_content[t[0]:t[1]] for t in reconst_target_locs]
            original_target_texts = [tid_to_text[tid] for tid in tids]
            for t1, t2 in zip(reconst_target_texts, original_target_texts):
                t1 = t1.strip()
                t2 = t2.strip()
                assert(t1==t2)

        split_text = [
            "\?", 
            "!", 
            "。", 
            "？", 
            "\n", 
            "\r", 
            "\[SEP\]"
        ]

        keep_before_text = [
            "[SEP]", 
            # "=Shared Post="
        ]

        split_pttn = "({0})".format("|".join(split_text))
        sents = re.split(split_pttn, self.data_dict["content"])

        tid_to_sid = {}
        tid_to_locs = {}
        tid_to_text = {}
        tid_to_neighbor = {}

        for tid, (start_idx, end_idx) in enumerate(self.data_dict["target_locs"]):
            tid_to_text[tid] = self.data_dict["content"][start_idx:end_idx]
            sid, start_idx, end_idx = find_idices_after_split(
                start_idx, end_idx, sents
            )
            prev_sids, next_sids = neighbor_sentences(
                sid, sents, n_prev=n_prev, n_next=n_next, split_text=split_text
            )
            tid_to_sid[tid] = sid
            tid_to_locs[tid] = (start_idx, end_idx)
            tid_to_neighbor[tid] = (prev_sids, next_sids)

        keep_before_idx = get_keep_before_idx(self.data_dict["content"], keep_before_text)
        reconst_sids = get_base_content(keep_before_idx, sents, split_text)

        for tid in tid_to_locs:
            prev_sids, next_sids = tid_to_neighbor[tid]

            reconst_sids.extend(prev_sids)
            reconst_sids.append(tid_to_sid[tid])
            reconst_sids.extend(next_sids)

        reconst_sids = sorted(list(set(reconst_sids)))
        reconst_content = "".join([sents[i] for i in reconst_sids])
        reconst_target_locs, tids = reconstruct_target_locs(reconst_sids, sents, tid_to_sid, tid_to_locs)
        # validate_target_locs(reconst_content, reconst_target_locs, tids, tid_to_text)

        self.data_dict["content"] = reconst_content
        self.data_dict["target_locs"] = reconst_target_locs
