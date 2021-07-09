import torch 


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
        # self.logits = self.predict(input_ids=None)
        # self.predicted_ids = self.logits.argmax(-1) 
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

        for idx, l in enumerate(lengths.tolist()):
            k = int(l * portion) 
            sorted_idxs = self.scores[idx, :].argsort()
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
        for p in range(0, 5):
            p = p / 10 
            input_ids = self.mask_tokens(portion=p, mode='sufficiency')
            batch = self.replace_input_ids(input_ids=input_ids)
            batches.append(batch)

        batch = self.concat_batches(batches)

        logits = self.predict(batch=batch)
        logits = torch.nn.functional.softmax(logits, dim=-1)

        predicted_cls = logits[:batch_size].argmax(-1)

        for idx, cls_id in enumerate(predicted_cls.tolist()):
            tmp = []
            for b in range(1, 6):
                tmp.append(logits[idx, cls_id].item() - logits[b * idx, cls_id].item())
                sufficiency.append(tmp)
        return sufficiency

    def get_comprehensiveness(self):
        # https://arxiv.org/pdf/1911.03429.pdf
        
        comprehensiveness = []
        batch_size = self.batch['input_ids'].size()[0]
        batches = [self.batch]
        for p in range(0, 5):
            p = p / 10 
            input_ids = self.mask_tokens(portion=p, mode='comprehensiveness')
            batch = self.replace_input_ids(input_ids=input_ids)
            batches.append(batch)

        batch = self.concat_batches(batches)

        logits = self.predict(batch=batch)
        logits = torch.nn.functional.softmax(logits, dim=-1)

        predicted_cls = logits[:batch_size].argmax(-1)

        for idx, cls_id in enumerate(predicted_cls.tolist()):
            tmp = []
            for b in range(1, 6):
                tmp.append(logits[idx, cls_id].item() - logits[b * idx, cls_id].item())
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
            _, _, logits = self.model(**inputs)
        return logits
