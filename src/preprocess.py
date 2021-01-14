# coding=utf-8
import re
from utils import SPEC_TOKEN


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
