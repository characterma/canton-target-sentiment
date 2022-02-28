import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd 


import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd 


class Comprehensiveness:
    def __init__(self, model, inputs, scores, unk_token_id, pad_token_id):
        """
        """
        self.model = model
        self.inputs = inputs
        self.scores = scores
        outputs = self.predict(inputs=self.inputs)
        self.predicted_cls = outputs["prediction"]
        self.probabilities = self.get_softmax_logits(
            logits=outputs["logits"],
        )
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.device = self.inputs["input_ids"].device
        self.run()

    def run(self):
        #  Test case
        self.sorted_idxs, self.sorted_scores, self.lengths = self.get_sorted_idxs()
        self.total_scores = [np.sum(s) for s in self.sorted_scores]
        self.masked_scores = self.get_masked_scores()
        self.comprehensiveness = self.get_comprehensiveness()

    def get_sorted_idxs(self):
        output1 = []
        output2 = []
        output3 = []
        lengths = self.inputs["attention_mask"].sum(1)
        
        for idx, l in enumerate(lengths.tolist()):
            scores_ = self.scores[idx, :l].tolist()
            scores = []
            exclude_idx = [l-1, 0]
            for i, s in enumerate(scores_):
                if i not in exclude_idx:
                    scores.append([i, s])
            scores = sorted(scores, key=lambda x: -x[1])
            sorted_idxs = [s[0] for s in scores]
            sorted_scores = [s[1] for s in scores]
            # sorted_idxs = torch.tensor(sorted_idxs).to(self.scores.device).long()
            output1.append(sorted_idxs)
            output2.append(sorted_scores)
            output3.append(len(sorted_scores))
        return output1, output2, output3

    def get_softmax_logits(self, logits):
        return F.softmax(logits, -1)

    def get_amended_tokens(self, to_mask, idx):
        max_length = self.inputs["input_ids"].size()[-1]
        to_keep = [i for i in range(max_length) if i not in to_mask]
        input_ids = torch.clone(self.inputs["input_ids"][idx, :])
        input_ids = input_ids[to_keep].tolist()
        input_ids = input_ids + [self.pad_token_id] * max(0, max_length - len(input_ids))
        attention_mask = torch.clone(self.inputs["attention_mask"][idx, :])
        attention_mask = attention_mask[to_keep].tolist()
        attention_mask = attention_mask + [0] * max(0, max_length - len(attention_mask))
        input_ids = torch.Tensor(input_ids).to(self.device).long()
        attention_mask = torch.Tensor(attention_mask).to(self.device).long()
        sample = self.replace_feature({"input_ids": input_ids, "attention_mask": attention_mask}, idx)
        return sample

    def mask_top_k(self, idx, k):
        """
        Args:
            idx: int
            k: int
        """
        to_mask = self.sorted_idxs[idx][:k]
        keep_score = np.sum(self.sorted_scores[idx][k:])
        mask_score = np.sum(self.sorted_scores[idx][:k])
        sample = self.get_amended_tokens(to_mask, idx)
        return sample, keep_score, mask_score, to_mask

    def concat_batches(self, batches):
        output = dict()
        for b in batches:
            for col in b:
                if col in output:
                    output[col].append(b[col])
                else:
                    output[col] = [b[col]]
        for col in output:
            output[col] = torch.cat(output[col], dim=0)
        return output

    def get_masked_scores(self):
        output = []
        for idx, cls_id in enumerate(self.predicted_cls):
            tmp = []
            sz = max(int(self.lengths[idx] / 10), 1)
            ks = []
            batches = []
            for k in range(sz, self.lengths[idx], sz):
                sample, keep_score, mask_score, mask_pos = self.mask_top_k(idx=idx, k=k)
                ks.append(k)
                batches.append(sample)
            logits = self.predict(inputs=self.concat_batches(batches))["logits"]
            probabilities = self.get_softmax_logits(logits=logits)
            for i, k in enumerate(ks):
                tmp.append(
                    {
                        "k": k, 
                        "mask_top_k_input_ids": sample["input_ids"].tolist(), 
                        "mask_top_k_keep_score": keep_score, 
                        "mask_top_k_mask_score": mask_score, 
                        "mask_top_k_mask_idxs": mask_pos, 
                        "mask_top_k_probabilities": probabilities[i, :].tolist(), 
                    }
                )
            output.append(tmp)
        return output

    def get_comprehensiveness(self):
        comprehensiveness = []
        for idx, cls_id in enumerate(self.predicted_cls):
            tmp = []
            for i in range(len(self.masked_scores[idx])):
                row = self.masked_scores[idx][i]
                tmp.append(self.probabilities[idx, cls_id].tolist() - row['mask_top_k_probabilities'][cls_id])
            comprehensiveness.append(tmp)
        return comprehensiveness

    def replace_feature(self, key_value_pairs, idx):
        batch = dict()
        for col in self.inputs:
            if col in key_value_pairs.keys():
                batch[col] = key_value_pairs[col].unsqueeze(0)
            else:
                batch[col] = self.inputs[col].index_select(0, torch.tensor([idx], device=self.device))
        return batch

    def predict(self, inputs):
        self.model.eval()
        outputs = self.model(**inputs)
        return outputs


class Faithfulness:
    def __init__(self, model, inputs, scores, unk_token_id, pad_token_id, metric=None):
        """
        """
        self.model = model
        self.inputs = inputs
        self.scores = scores
        outputs = self.predict(inputs=self.inputs)
        self.predicted_cls = outputs["prediction"]
        self.probabilities = self.get_softmax_logits(
            logits=outputs["logits"],
        )
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.metric = metric
        self.device = self.inputs["input_ids"].device
        self.run()

    def run(self):
        #  Test case
        self.sorted_idxs, self.sorted_scores, self.lengths = self.get_sorted_idxs()
        self.total_scores = [np.sum(s) for s in self.sorted_scores]
        self.masked_scores = self.get_masked_scores()
        self.sufficiency, self.comprehensiveness = self.get_sufficiency_and_comprehensiveness()
        self.decision_flip_mit = self.get_decision_flip_mit()
        self.decision_flip_fot = self.get_decision_flip_fot()
        self.importance_probability_correlation = self.get_importance_probability_correlation()
        self.monotonicity = self.get_monotonicity()

    def get_sorted_idxs(self):
        output1 = []
        output2 = []
        output3 = []
        lengths = self.inputs["attention_mask"].sum(1)
        
        for idx, l in enumerate(lengths.tolist()):
            scores_ = self.scores[idx, :l].tolist()
            scores = []
            exclude_idx = [l-1, 0]
            for i, s in enumerate(scores_):
                if i not in exclude_idx:
                    scores.append([i, s])
            scores = sorted(scores, key=lambda x: -x[1])
            sorted_idxs = [s[0] for s in scores]
            sorted_scores = [s[1] for s in scores]
            # sorted_idxs = torch.tensor(sorted_idxs).to(self.scores.device).long()
            output1.append(sorted_idxs)
            output2.append(sorted_scores)
            output3.append(len(sorted_scores))
        return output1, output2, output3

    def get_softmax_logits(self, logits):
        return F.softmax(logits, -1)

    def get_amended_tokens(self, to_mask, idx):
        max_length = self.inputs["input_ids"].size()[-1]
        to_keep = [i for i in range(max_length) if i not in to_mask]
        input_ids = torch.clone(self.inputs["input_ids"][idx, :])
        input_ids = input_ids[to_keep].tolist()
        input_ids = input_ids + [self.pad_token_id] * max(0, max_length - len(input_ids))
        attention_mask = torch.clone(self.inputs["attention_mask"][idx, :])
        attention_mask = attention_mask[to_keep].tolist()
        attention_mask = attention_mask + [0] * max(0, max_length - len(attention_mask))
        input_ids = torch.Tensor(input_ids).to(self.device).long()
        attention_mask = torch.Tensor(attention_mask).to(self.device).long()
        sample = self.replace_feature({"input_ids": input_ids, "attention_mask": attention_mask}, idx)
        return sample

    def mask_k_th(self, idx, k):
        """
        Args:
            idx: int
            k: int
        """
        to_mask = [self.sorted_idxs[idx][k-1]]
        keep_score = self.sorted_scores[idx][k-1]
        mask_score = self.total_scores[idx] - keep_score
        sample = self.get_amended_tokens(to_mask, idx)
        return sample, keep_score, mask_score, to_mask

    def keep_k_th(self, idx, k):
        """
        Args:
            idx: int
            k: int
        """
        to_mask = [i for i in self.sorted_idxs[idx] if i!=self.sorted_idxs[idx][k-1]]
        mask_score = self.sorted_scores[idx][k-1]
        keep_score = self.total_scores[idx] - mask_score
        sample = self.get_amended_tokens(to_mask, idx)
        return sample, keep_score, mask_score, to_mask

    def mask_top_k(self, idx, k):
        """
        Args:
            idx: int
            k: int
        """
        to_mask = self.sorted_idxs[idx][:k]
        keep_score = np.sum(self.sorted_scores[idx][k:])
        mask_score = np.sum(self.sorted_scores[idx][:k])
        sample = self.get_amended_tokens(to_mask, idx)
        return sample, keep_score, mask_score, to_mask

    def keep_top_k(self, idx, k):
        """
        Args:
            idx: int
            k: int
        return:
            input_ids: [L]
        """
        to_mask = self.sorted_idxs[idx][k:]
        keep_score = np.sum(self.sorted_scores[idx][:k])
        mask_score = np.sum(self.sorted_scores[idx][k:])
        sample = self.get_amended_tokens(to_mask, idx)
        return sample, keep_score, mask_score, to_mask

    def concat_batches(self, batches):
        output = dict()
        for b in batches:
            for col in b:
                if col in output:
                    output[col].append(b[col])
                else:
                    output[col] = [b[col]]
        for col in output:
            output[col] = torch.cat(output[col], dim=0)
        return output

    def get_masked_scores(self):
        output = []
        for idx, cls_id in enumerate(self.predicted_cls):
            tmp = []
            for k in range(1, self.lengths[idx] + 1):
                batches = []
                sample_0, keep_score_0, mask_score_0, mask_pos_0 = self.mask_k_th(idx=idx, k=k)
                batches.append(sample_0)
                sample_1, keep_score_1, mask_score_1, mask_pos_1 = self.keep_k_th(idx=idx, k=k)
                batches.append(sample_1)
                sample_2, keep_score_2, mask_score_2, mask_pos_2 = self.mask_top_k(idx=idx, k=k)
                batches.append(sample_2)
                sample_3, keep_score_3, mask_score_3, mask_pos_3 = self.keep_top_k(idx=idx, k=k)
                batches.append(sample_3)
                logits = self.predict(inputs=self.concat_batches(batches))["logits"]
                probabilities = self.get_softmax_logits(logits=logits)
                tmp.append(
                    {
                        "k": k, 
                        "mask_k_th_input_ids": sample_0["input_ids"].tolist(), 
                        "mask_k_th_keep_score": keep_score_0, 
                        "mask_k_th_mask_score": mask_score_0, 
                        "mask_k_th_mask_idxs": mask_pos_0, 
                        "mask_k_th_probabilities": probabilities[0, :].tolist(), 
                        "keep_k_th_input_ids": sample_1["input_ids"].tolist(), 
                        "keep_k_th_keep_score": keep_score_1, 
                        "keep_k_th_mask_score": mask_score_1, 
                        "keep_k_th_mask_idxs": mask_pos_1, 
                        "keep_k_th_probabilities": probabilities[1, :].tolist(), 
                        "mask_top_k_input_ids": sample_2["input_ids"].tolist(), 
                        "mask_top_k_keep_score": keep_score_2, 
                        "mask_top_k_mask_score": mask_score_2, 
                        "mask_top_k_mask_idxs": mask_pos_2, 
                        "mask_top_k_probabilities": probabilities[2, :].tolist(), 
                        "keep_top_k_input_ids": sample_3["input_ids"].tolist(), 
                        "keep_top_k_keep_score": keep_score_3, 
                        "keep_top_k_mask_score": mask_score_3, 
                        "keep_top_k_mask_idxs": mask_pos_3, 
                        "keep_top_k_probabilities": probabilities[3, :].tolist(), 
                    }
                )
            output.append(tmp)
        return output

    def get_sufficiency_and_comprehensiveness(self):
        sufficiency = []
        comprehensiveness = []
        for idx, cls_id in enumerate(self.predicted_cls):
            tmp_1 = []
            tmp_2 = []
            for p in range(1, 11):
                p = p / 10
                row = self.masked_scores[idx][int(p * self.lengths[idx]) - 1]
                tmp_1.append(self.probabilities[idx, cls_id].tolist() - row['keep_top_k_probabilities'][cls_id])
                tmp_2.append(self.probabilities[idx, cls_id].tolist() - row['mask_top_k_probabilities'][cls_id])
            sufficiency.append(tmp_1)
            comprehensiveness.append(tmp_2)
        return sufficiency, comprehensiveness

    def get_decision_flip_mit(self):
        # https://arxiv.org/pdf/2105.02657.pdf
        dec_flip_mit = []
        for idx, c0 in enumerate(self.predicted_cls):
            c1 = np.argmax(
                self.masked_scores[idx][0]["mask_k_th_probabilities"]
            )
            if c0 != c1:
                dec_flip_mit.append(1)
            else:
                dec_flip_mit.append(0)
        return dec_flip_mit

    def get_decision_flip_fot(self):
        # https://arxiv.org/pdf/2105.02657.pdf
        dec_flip_fot = []
        for idx, c0 in enumerate(self.predicted_cls):
            for k in range(self.lengths[idx]):
                c1 = np.argmax(
                    self.masked_scores[idx][k]["mask_top_k_probabilities"]
                )
                if c1 != c0 or k == self.lengths[idx]-1:
                    dec_flip_fot.append((k + 1) / self.lengths[idx])
                    break  # inner loop
        return dec_flip_fot

    def get_importance_probability_correlation(self):
        correlations = []
        for idx, predicted_cls in enumerate(self.predicted_cls):
            probabilities = []
            importances = []
            for ms in self.masked_scores[idx]:
                probabilities.append(
                    ms["keep_k_th_probabilities"][predicted_cls]
                )
                importances.append(
                    ms["keep_k_th_mask_score"]
                )
            correlations.append(
                np.corrcoef(probabilities, importances)[0, 1]
            )
        return correlations
            
    def get_monotonicity(self):
        correlations = []
        for idx, predicted_cls in enumerate(self.predicted_cls):
            probabilities = []
            importances = []
            for ms in self.masked_scores[idx]:
                probabilities.append(
                    ms["mask_top_k_probabilities"][predicted_cls]
                )
                importances.append(
                    ms["mask_top_k_mask_score"]
                )
            correlations.append(
                np.corrcoef(probabilities, importances[-1::-1])[0, 1]
            )
        return correlations
            
    def replace_feature(self, key_value_pairs, idx):
        batch = dict()
        for col in self.inputs:
            if col in key_value_pairs.keys():
                batch[col] = key_value_pairs[col].unsqueeze(0)
            else:
                batch[col] = self.inputs[col].index_select(0, torch.tensor([idx], device=self.device))
        return batch

    def predict(self, inputs):
        self.model.eval()
        outputs = self.model(**inputs)
        return outputs
