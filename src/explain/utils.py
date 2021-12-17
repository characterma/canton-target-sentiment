from captum.attr import visualization as viz
import numpy as np


def get_explanation(pipeline, raw_data, enable_faithfulness=False):
    tokens, scores, attr_target, attr_target_prob, tokens_encoded, comprehensiveness = pipeline.explain(
        data_dict=raw_data,
        method='IntegratedGradients',
        layer='pretrained_model.embeddings.word_embeddings', 
        norm='sum',
        enable_faithfulness=enable_faithfulness  # use comp if True
    )
    return tokens, scores, attr_target, attr_target_prob, tokens_encoded, comprehensiveness
    

def visualize_data_record_bert(pipeline, raw_data, tokens, scores, attr_target, attr_target_prob):
    if raw_data.get('label') is not None:
        true_class = raw_data['label']
    else:
        true_class = None
    attr_class = pipeline.args.label_to_id_inv[attr_target]
    start_position_vis = viz.VisualizationDataRecord(
                            scores,
                            pred_prob=attr_target_prob,
                            pred_class=attr_class,
                            true_class=true_class,
                            attr_class=attr_class,
                            attr_score=np.sum(scores),       
                            raw_input=tokens,
                            convergence_score=None)
    return viz.visualize_text([start_position_vis])
    
    
def get_max_magnitude(arr):
    return arr[np.argmax(np.absolute(arr))]

    
def get_segment_level_explanation(seg_func, raw_text, tokens, scores, tokens_encoded):
    """
    Args:
        scores: token level scores
    """
    
    # to char level scores
    scores_char = []
    for char_id in range(len(raw_text)):
        token_id = tokens_encoded.char_to_token(char_id)
        if token_id is not None:
            scores_char.append((token_id, scores[token_id]))
        else:
            scores_char.append((token_id, 0))
    
    # get segments indexs
    segments, segments_idxs = seg_func(raw_text)
    
    # seg level scores
    scores_seg = []
    cur_idx = 0
    for idx in segments_idxs:
        scores_ = scores_char[idx[0]:idx[1]]
        scores_ = [x[-1] for x in list(set(scores_))]
        scores_seg.append(
            np.sum(scores_)
        )
    
    return segments, scores_seg
        