# -*- coding: utf-8 -*-


def get_token_level_tags(tokens_encoded, sent_indexs, postags, scheme="BI"):
    token_tags = dict()
    for tag, (start_idx, end_idx) in zip(postags, sent_indexs):

        token_idx_list = []
        for char_idx in range(start_idx, end_idx):
            token_idx = tokens_encoded.char_to_token(char_idx)
            if token_idx is not None:
                token_idx_list.append(token_idx)

        token_idx_list = sorted(set(token_idx_list))
        if len(token_idx_list) > 0:
            token_tags[token_idx_list[0]] = "B-" + tag
            for token_idx in token_idx_list[1:]:
                token_tags[token_idx] = "I-" + tag

    output = []
    for token_idx in range(tokens_encoded.length[0]):
        if token_idx in token_tags:
            output.append(token_tags[token_idx])
        else:
            output.append("O")

    return output