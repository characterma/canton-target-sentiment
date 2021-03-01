
import os, sys
import pickle
import lime
import torch
import pandas as pd
import numpy as np

from pathlib import Path
from lime.lime_text import LimeTextExplainer
from collections import defaultdict 


SPEC_TOKENS = ["[SEP]", "[CLS]"]


class Explanation:
    """
    Base class
    Usage:
        self.scores: [[0, token0, score0], [1, token1, score1], ...]
    """
    def __init__(self, model, tokenizer, features, batch_size=32, faithfulness=False, pred_probs=None, pred_cls=None):
        assert("tokens" in features)
        self.scores = None
        self.model = model
        self.model.eval()
        self.features = features
        self.tokenizer = tokenizer
        self.tokens = features['tokens']
        self.pad_tokens = []
        self.non_pad_tokens = []
        for tkn in self.tokens:
            if tkn!="[PAD]":
                self.non_pad_tokens.append(tkn)
            else:
                self.pad_tokens.append(tkn)
        self.batch_size = batch_size

        if faithfulness:
            if pred_probs is None or pred_cls is None:
                self.pred_probs = self.predict([" ".join(self.non_pad_tokens)])
                self.pred_cls = self.pred_probs.argmax()
            else:
                self.pred_probs = pred_probs
                self.pred_cls = pred_cls

            self.total_scores, self.comprehensiveness, self.sufficiency, self.comprehensiveness_input, self.sufficiency_input = self.get_faithfulness()
        else:
            self.total_scores = None 
            self.comprehensiveness = None 
            self.sufficiency = None 
            self.comprehensiveness = None 
            self.comprehensiveness_input = None 
            self.sufficiency = None 
            self.sufficiency_input = None 

    def get_masked_text(self, total_score=0.5, keep=False, return_tokens=False):
        if self.scores is None:
            return None

        n_tokens = int(top_percent * len(self.scores))
        scores_desc = sorted(self.scores, key=lambda x: x[2], reverse=True)

        cul_score = 0

        for i in range(len(scores_desc)):
            if keep and cul_score > total_score:
                scores_desc[i][1] = "[UNK]"
            elif not keep and cul_score <= total_score:
                scores_desc[i][1] = "[UNK]"
            cul_score += scores_desc[i][2]

        scores = sorted(scores_desc, key=lambda x: x[0])

        if return_tokens:
            return [x[1] for x in scores]
        else:
            return " ".join([x[1] for x in scores])

    def predict(self, masked_texts):
        """
        Args:
            masked_texts: list of masked texts.
        Return:
            probabilities [N, C], where N is the number of samples, C is the number of class.
        """
        outputs = []
        for i in range(0, len(masked_texts), self.batch_size):
            inputs = defaultdict(list)
            for text in masked_texts[i:i + batch_size]:
                with torch.no_grad():
                    for col in self.model.INPUT_COLS:
                        if col!="raw_text" and col in self.features:
                            inputs[col].append(self.features[col])
                        elif col=="raw_text":
                            token_ids = torch.tensor(
                                self.tokenizer.convert_tokens_to_ids(text.split(" ") + self.pad_tokens)
                            )
                            inputs["raw_text"].append(token_ids)
                    
            for key, value in inputs.items():
                inputs[key] = torch.stack(inputs[key], dim=0).to(self.model.device)

            results = self.model(**inputs)
            probs = torch.nn.functional.softmax(results[1], dim=1)
            probs = probs.detach().cpu().numpy()
            outputs.append(probs)
                    
        return np.concatenate(outputs, axis=0)
    
    def get_faithfulness(self):
        comprehensiveness_input = [] # for debug
        sufficiency_input = [] # for debug
        total_scores = []
        comprehensiveness = []
        sufficiency = []
        bins = np.array(range(1, 6)) / 10
        for score in bins:
            # comprehensiveness
            text = self.get_masked_text(total_score=score, keep=False, return_tokens=False)
            probs = self.predict([text])[0]
            comprehensiveness.append(self.pred_probs - probs[self.pred_cls])
            comprehensiveness_input.append(text)

            # sufficiency
            text = self.get_masked_text(total_score=score, keep=True, return_tokens=False)
            probs = self.predict([text])[0]
            sufficiency.append(self.pred_probs - probs[self.pred_cls])
            sufficiency_input.append(text)

            total_scores.append(score)

        return total_scores, comprehensiveness, sufficiency, comprehensiveness_input, sufficiency_input


class AttnExplanation(Explanation):
    """
    Usage:
        self.scores: [[0, token0, score0], [1, token1, score1], ...]
    """
    def __init__(
        self, 
        model, 
        tokenizer, 
        features, 
        pred_probs, 
        pred_cls, 
        attn_mat, 
        faithfulness=False, 
        batch_size=32, 
        exclude_spec_tokens=True
    ):
        super(AttnExplanation, self).__init__(
            model=model, 
            tokenizer=tokenizer, 
            features=features, 
            batch_size=batch_size, 
            faithfulness=faithfulness, 
            pred_probs=pred_probs, 
            pred_cls=pred_cls
        )
        assert('target_span' in features)
        self.target_span = features['target_span'].tolist()
        _scores = attn_mat[:, :, self.target_span[0]: self.target_span[1]+1, :].sum(axis=0).sum(axis=0)
        if exclude_spec_tokens:
            for ii, tkn in enumerate(self.tokens):
                if tkn in SPEC_TOKENS:
                    _scores[ii] = 0
        _scores = _scores / _scores.sum()
        
        self.scores = []
        for i, t, s in enumerate(zip(self.tokens, _scores)):
            if t != "[PAD]":
                self.scores.append([i, t, s])


class LimeExplanation(Explanation):

    def __init__(
        self, 
        model, 
        tokenizer
        features,  
        pred_probs, 
        pred_cls,
        num_samples=500, 
        non_negative=True, 
        exclude_spec_tokens=True, 
        batch_size=32 ,
        faithfulness=False
    ):
        super(LimeExplanation, self).__init__(
            model=model, 
            tokenizer=tokenizer, 
            features=features, 
            batch_size=batch_size, 
            faithfulness=faithfulness, 
            pred_probs=pred_probs, 
            pred_cls=pred_cls
        )

        self.lime_explainer = LimeTextExplainer(class_names=("neutral", "negative", "positive"), 
                              mask_string="[UNK]", 
                              bow=False)

        self.num_samples = num_samples
        self.exclude_spec_tokens = exclude_spec_tokens
        self.non_negative = non_negative

        self.spec_token_ids = []
        for i, t in enumerate(self.non_pad_tokens):
            if t in SPEC_TOKENS:
                self.spec_token_ids.append(i)

        self.lime_input = " ".join(self.non_pad_tokens)
        self.scores = self.get_lime_scores()
        
    def get_lime_scores(self):
        explanation = self.lime_explainer.explain_instance(
            self.lime_input, 
            self.predict,
            top_labels=1,
            num_samples=self.num_samples, 
            except_token_ids=self.spec_token_ids if self.exclude_spec_tokens else None
        )
        score_map = explanation.as_map()
        _scores = [0] * len(self.non_pad_tokens)
        for i, s in score_map[self.pred_cls]:
            _scores[i] = max(s , 0) if self.non_negative else s
        _scores = np.array(_scores)
        _scores = _scores / _scores.sum()
        return list(zip(range(len(self.non_pad_tokens)), self.non_pad_tokens, _scores))
        

    