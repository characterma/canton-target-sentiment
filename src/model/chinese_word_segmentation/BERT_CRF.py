import torch.nn as nn
from transformers import BertPreTrainedModel

from model.layer.crf import LinearChainCRF
from model.utils import load_pretrained_bert, load_pretrained_config, NLPModelOutput


class BERT_CRF(BertPreTrainedModel):
    def __init__(self, args):
        super(BERT_CRF, self).__init__(load_pretrained_config(args))

        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)
        if not args.model_config["embedding_trainable"]:
            self.freeze_emb()
        self.pretrained_config = load_pretrained_config(args)
        self.num_labels = len(args.label_to_id)

        self.bert_dropout = nn.Dropout(
            self.model_config["bert_dropout"]
        )
        self.crf = LinearChainCRF(self.pretrained_config.hidden_size, self.num_labels)
        self.return_logits = False
        self.to(args.device)

    def freeze_emb(self):
        # Freeze all parameters except self attention parameters
        for param_name, param in self.pretrained_model.named_parameters():
            if "embeddings" in param_name:
                param.requires_grad = False

    def set_return_logits(self):
        self.return_logits = True

    def forward(self, input_ids, attention_mask, label=None):
        outputs = dict()
        lm = self.pretrained_model(
            input_ids=input_ids.long(), attention_mask=attention_mask.long(), return_dict=True
        )
        logits = lm["last_hidden_state"]
        logits = self.bert_dropout(logits)

        if self.return_logits:
            return logits 
        else:
            prediction, scores = self.crf.viterbi_decode(
                logits, length_index=attention_mask
            )  # [B, 1, L], [B, 1]
            
            if label is not None:
                loss = self.crf.nll_loss(
                    x=logits, y=label, length_index=attention_mask, reduce="mean"
                )
            else:
                loss = None

            prediction = [p[0] for p in prediction]
            outputs = NLPModelOutput(
                loss=loss, 
                prediction=prediction, 
                logits=logits, 
            )
            return outputs