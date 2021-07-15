import torch 
import logging 


class Faithfulness:
    def __init__(self, model, batch, scores, mask_id, args):
        """
        model: outputs logits
        batch: model inputs, [B, L]
        scores: [B, L]
        """
        self.args = args 
        self.model = model 
        self.batch = batch 
        self.scores = scores
        self.predicted_cls = self.predict(batch=self.batch).argmax(-1)
        self.mask_id = mask_id

        # run faithfulness
        self.sufficiency = self.get_sufficiency()
        self.comprehensiveness = self.get_comprehensiveness()

    def mask_tokens(self, portion, mode):
        """
        portion: value in (0, 1), [0.1, 0.2, ...]
        mode: "sufficiency" or "comprehensiveness"
        """
        input_ids = torch.clone(self.batch['input_ids'])
        lengths = self.batch['attention_mask'].sum(1)

        # print(input_ids)
        for idx, l in enumerate(lengths.tolist()):
            k = int(l * portion) 
            sorted_idxs = self.scores[idx, :l].argsort(descending=True)

            if mode=='comprehensiveness':
                to_mask = sorted_idxs[:k]
                input_ids[idx, to_mask] = self.mask_id
            elif mode=="sufficiency": 
                to_mask = sorted_idxs[k:l]
                input_ids[idx, to_mask] = self.mask_id
            else:
                raise(ValueError)

        return input_ids # [B, L]


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


    def get_sufficiency(self):
        sufficiency = []
        batch_size = self.batch['input_ids'].size()[0]
        batches = [self.batch]
        for p in range(1, 6):
            p = p / 10 
            input_ids = self.mask_tokens(portion=p, mode='sufficiency')
            batch = self.replace_input_ids(input_ids=input_ids)
            batches.append(batch)

        batch = self.concat_batches(batches) # [B0,B1,B2,..B5]

        logits = self.predict(batch=batch)
        logits = torch.nn.functional.softmax(logits, dim=-1)

        for idx, cls_id in enumerate(self.predicted_cls.tolist()):
            tmp = []
            logit0 = logits[idx, cls_id].item()
            for i in range(0, 6):
                logit1 = logits[i * batch_size + idx, cls_id].item()
                tmp.append(logit0 - logit1)
            sufficiency.append(tmp)
        return sufficiency

    def get_comprehensiveness(self):
        # https://arxiv.org/pdf/1911.03429.pdf
        
        comprehensiveness = []
        batch_size = self.batch['input_ids'].size()[0]
        batches = [self.batch]
        for p in range(1, 6):
            p = p / 10 
            input_ids = self.mask_tokens(portion=p, mode='comprehensiveness')
            batch = self.replace_input_ids(input_ids=input_ids)
            batches.append(batch)

        batch = self.concat_batches(batches)

        logits = self.predict(batch=batch)
        logits = torch.nn.functional.softmax(logits, dim=-1)

        for idx, cls_id in enumerate(self.predicted_cls.tolist()):
            tmp = []
            logit0 = logits[idx, cls_id].item()
            for i in range(0, 6):
                logit1 = logits[i * batch_size + idx, cls_id].item()
                tmp.append(logit0 - logit1)
            comprehensiveness.append(tmp)

        return comprehensiveness

    def replace_input_ids(self, input_ids):
        batch = dict()
        for col in self.batch:
            if col=='input_ids' and input_ids is not None:
                batch[col] = input_ids
            else:
                batch[col] = self.batch[col]
        return batch

    def predict(self, batch):
        self.model.eval()
        inputs = dict()
        with torch.no_grad():
            for col in batch:
                if torch.is_tensor(batch[col]):
                    inputs[col] = batch[col].to(self.args.device).long()
            outputs = self.model(**inputs)
            logits = outputs[2]
        return logits
