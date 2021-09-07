import torch 
import torch.nn.functional as F


class WordOmission:
    def __init__(self, model):
        self.model = model

    def get_softmax_logits(self, logits):
        return F.softmax(logits, -1)

    def attribute(self, input_ids, attention_mask, target):
        scores = []
        logits0 = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        logits0 = self.get_softmax_logits(logits=logits0)
        prob0 = logits0[0, target].item()

        length = attention_mask.sum(-1).item()
        max_length = input_ids.shape[-1]
        for i in range(length):
            if i==0 or i>=length-1:
                scores.append(0)
            else:
                mask = torch.ones_like(attention_mask)
                mask[0, i] = 0
                input_ids_i = input_ids[mask==1]
                attention_mask_i = attention_mask[mask==1]
                cur_length = input_ids_i.shape[-1]
                input_ids_i = F.pad(input_ids_i, (0, max_length - cur_length), "constant", 0)
                attention_mask_i = F.pad(attention_mask_i, (0, max_length - cur_length), "constant", 0)
                logitsi = self.model(
                    input_ids=input_ids_i.unsqueeze(0), 
                    attention_mask=attention_mask_i.unsqueeze(0)
                )
                logitsi = self.get_softmax_logits(logits=logitsi)
                probi = logitsi[0, target].item()
                scores.append(prob0 - probi)
        return torch.Tensor([scores]).to(input_ids.device)






