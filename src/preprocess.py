# coding=utf-8
import re
import numpy as np
from utils import SPEC_TOKEN
from hanziconv import HanziConv


def standardize_text(raw_text):
    raw_text = raw_text.lower()
    raw_text = HanziConv.toSimplified(raw_text)
    return raw_text

def emoji_index_conversion(text, indices):
    # indices: list of list
    python_index = []
    java_index = []
    incident = 0
    for i in range(len(text)):
        python_index.append(i)
        java_index.append(i + incident)
        if len(text[i].encode('utf-8')) == 4:
            incident += 1

    python_index.append(len(text))
    java_index.append(len(text) + incident)

    _idx_map = list(zip(java_index, python_index))
    _idx_map = sorted(_idx_map, key=lambda x: x[0])
    idx_map = dict()
    for i in range(len(_idx_map)):
        if i < len(_idx_map) - 1:
            if _idx_map[i][0] + 1 < _idx_map[i+1][0]:
                for j in range(_idx_map[i][0] + 1, _idx_map[i+1][0]):
                    idx_map[j] = _idx_map[i][1]

        idx_map[_idx_map[i][0]] = _idx_map[i][1]

    for x in indices:
        if x[0] > len(text):
            x[0] = x[0] + incident
        else:
            x[0] = idx_map[x[0]]

        if x[1] > len(text):
            x[1] = x[1] + incident
        else:
            x[1] = idx_map[x[1]]
    return indices


def get_hl_content_spans(
    text, hl_separator="## Headline ##", ct_separator="## Content ##"
):
    """"""
    hl_st_idx = None
    hl_ed_idx = None
    ct_st_idx = None
    # if "## Headline ##" in text:
    hl_match = re.search(hl_separator, text)
    ct_match = re.search(ct_separator, text)

    if hl_match:
        _, hl_st_idx = hl_match.span()

    if ct_match:
        hl_ed_idx, ct_st_idx = ct_match.span()

    return hl_st_idx, hl_ed_idx, ct_st_idx


def clean_text_hk_beauty(text, st_indices=None, ed_indices=None):
    patterns = [
        r"原帖由.{3,}發表",
        r"引用:",
        r"## Headline ##",
        r"## Content ##",
    ]
    text_clean = text
    for p in patterns:
        match = re.search(p, text_clean)
        # print("***", st_indices)
        if match:
            # print(match.span(), indices)
            st_idx, ed_idx = match.span()
            # print(text_clean[st_idx:ed_idx])
            text_clean = text_clean[:st_idx] + text_clean[ed_idx:]

            if st_indices is not None:
                for i in range(len(st_indices)):
                    if st_indices[i] is None:
                        continue
                    if st_idx <= st_indices[i] < ed_idx - 1:
                        st_indices[i] = None
                    elif ed_idx - 1 <= st_indices[i]:
                        st_indices[i] = st_indices[i] - (ed_idx - st_idx)

            if ed_indices is not None:
                for i in range(len(ed_indices)):
                    if ed_indices[i] is None:
                        continue
                    if st_idx + 1 <= ed_indices[i] <= ed_idx:
                        ed_indices[i] = None
                    elif ed_idx < ed_indices[i]:
                        ed_indices[i] = ed_indices[i] - (ed_idx - st_idx)

    return text_clean, st_indices, ed_indices


def truncate_by_target_center(
    tgt_st_idx,
    tgt_ed_idx,
):
    pass

def preprocess_text_hk_gov(content, target_indices, names=None, n_prev=0, n_next=0):
    """
    Return:
        reconst_content, reconst_target_indices, ""
    """
    def search_for_idx(name, content, st_idx_old):
        matches = [m.span()[0] for m in re.finditer(name, content)]
        if not matches:
            return None
        dist = np.absolute(np.array(matches) - st_idx_old)
        st_idx_new = matches[np.argmin(dist)]
        return st_idx_new

    # target_indices = emoji_index_conversion(content, target_indices)
    for t, nme in zip(target_indices, names):
        if content[t[0]:t[1]]!=nme:
            start_idx = search_for_idx(nme, content, t[0])
            if start_idx is None:
                return None, None, "start_idx is None"
            end_idx = start_idx + len(nme)
            t[0] = start_idx
            t[1] = end_idx

    sents = re.split("\?|!|。|？|！|\n|\r", content)
    
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

    paragraphs = []

    for start_idx, end_idx in target_indices:
        
        sentence_id, start_idx, end_idx = find_idices_after_split(
            start_idx,
            end_idx,
            sents
        )
        
        if sentence_id is None:
            return None, None, "sentence_id is None"
        prev, next = neighbor_sentences(sentence_id, sents, n_prev=n_prev, n_next=n_next)
        paragraphs.append({
            'prev': prev, 
            'next': next,
            'sentence_id': sentence_id,
            'start_idx': start_idx,
            'end_idx': end_idx,
        })
    
    reconst_sent_ids = []
    reconst_target_indices = []
    curr_len = 0
    for s in paragraphs:
        start_idx = s['start_idx'] + curr_len
        end_idx = s['end_idx'] + curr_len
        for idx in s['prev']:
            if idx not in reconst_sent_ids and idx != s['sentence_id']:
                reconst_sent_ids.append(idx)
                start_idx += len(sents[idx])
                end_idx += len(sents[idx])
                curr_len += len(sents[idx])
                
        reconst_sent_ids.append(s['sentence_id'])
        curr_len += len(sents[s['sentence_id']])
        
        for idx in s['next']:
            if idx not in reconst_sent_ids and idx != s['sentence_id']:
                reconst_sent_ids.append(idx)
                curr_len += len(sents[idx])
        reconst_target_indices.append([start_idx, end_idx])
        
    reconst_content = "".join([sents[i] for i in reconst_sent_ids])
    
    if names is not None:
        assert(len(names)==len(reconst_target_indices))
        for t, nme in zip(reconst_target_indices, names):
            if not reconst_content[t[0]:t[1]]==nme:
                return None, None, "reconst_content[t[0]:t[1]]!=nme"

    return reconst_content, reconst_target_indices, ""
        

def preprocess_text_hk_beauty(
    text, tgt_st_idx, tgt_ed_idx, sent_sep="[SEP]", num_prev_sents=15, num_next_sents=15
):
    hl_st_idx, hl_ed_idx, ct_st_idx = get_hl_content_spans(text)
    (
        text,
        [hl_st_idx, ct_st_idx, tgt_st_idx],
        [hl_ed_idx, tgt_ed_idx],
    ) = clean_text_hk_beauty(
        text, [hl_st_idx, ct_st_idx, tgt_st_idx], [hl_ed_idx, tgt_ed_idx]
    )

    if tgt_st_idx is None or tgt_ed_idx is None:
        return "", (None, None), "", "", "", ""

    # For target in headline
    if hl_st_idx <= tgt_st_idx and tgt_ed_idx <= hl_ed_idx:
        tgt_in_hl = True
        # In this case, target sentence == headline
        tgt_sent = text[hl_st_idx:hl_ed_idx]
        tgt_st_idx = tgt_st_idx - hl_st_idx
        tgt_ed_idx = tgt_ed_idx - hl_st_idx

        # Append the last n sentences in the post
        sentences = [
            s.strip() for s in text[hl_ed_idx:].split("\n") if len(s.strip()) > 0
        ]
        next_sents = sentences[-(num_prev_sents + num_next_sents) :]
        if len(next_sents) > 0:
            next_sents = sent_sep.join(next_sents)
        else:
            next_sents = ""
        return tgt_sent, (tgt_st_idx, tgt_ed_idx), "", "", next_sents, tgt_in_hl

    else:  # For target in content
        tgt_in_hl = False
        cur_idx = 0
        tgt_sent_idx = None

        # Retrieve headline
        hl_sent = text[hl_st_idx:hl_ed_idx].strip()

        # Exclude headline, split sentences and shift target indices
        sentences = text[hl_ed_idx:].split("\n")
        tgt_st_idx = tgt_st_idx - hl_ed_idx
        tgt_ed_idx = tgt_ed_idx - hl_ed_idx

        # Find the index of target sentence, & the target span w.r.t the target sentence.
        for sent_idx, sent in enumerate(sentences):
            matched = ""
            if len(sent) == 0:
                cur_idx += 1
            else:
                for c in sent:
                    if tgt_st_idx <= cur_idx < tgt_ed_idx:
                        matched += c
                    cur_idx += 1
                cur_idx += 1

            if matched:
                tgt_sent_idx = sent_idx
                tgt_st_idx = sent.find(matched)
                tgt_ed_idx = tgt_st_idx + len(matched)
                break

        tgt_sent = ""
        prev_sents = []
        next_sents = []

        # Retrieve target sentence, previous sentences, and next sentences.
        if tgt_sent_idx is not None:
            tgt_sent = sentences[tgt_sent_idx]

            for s in sentences[tgt_sent_idx + 1 :]:
                if s != "" and len(next_sents) < num_next_sents:
                    next_sents.append(s)
                elif len(next_sents) >= num_next_sents:
                    break

            for s in sentences[:tgt_sent_idx][-1::-1]:
                if s != "" and len(prev_sents) < num_prev_sents:
                    prev_sents.append(s)
                elif len(prev_sents) >= num_prev_sents:
                    break

        # Separate sentences by [SEP]
        if len(prev_sents) > 0:
            prev_sents = sent_sep.join(prev_sents)
        else:
            prev_sents = ""

        if len(next_sents) > 0:
            next_sents = sent_sep.join(next_sents)
        else:
            next_sents = ""

        return (
            tgt_sent,
            (tgt_st_idx, tgt_ed_idx),
            hl_sent,
            prev_sents,
            next_sents,
            tgt_in_hl,
        )


def get_mask_target(tgt_sent, tgt_st_idx, tgt_ed_idx):
    tgt_sent = tgt_sent[:tgt_st_idx] + SPEC_TOKEN.TARGET + tgt_sent[tgt_ed_idx:]
    tgt_ed_idx = tgt_st_idx + len(SPEC_TOKEN.TARGET)
    return tgt_sent, (tgt_st_idx, tgt_ed_idx)
