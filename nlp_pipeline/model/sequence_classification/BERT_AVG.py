import torch
import torch.nn as nn
from transformers import BertPreTrainedModel

from nlp_pipeline.model.layer.fc import LinearLayer
from nlp_pipeline.model.utils import load_pretrained_bert, load_pretrained_config, NLPModelOutput


class BERT_AVG(BertPreTrainedModel):
    def __init__(self, args):
        super(BERT_AVG, self).__init__(load_pretrained_config(args))
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(args)
        if not args.model_config["embedding_trainable"]:
            self.freeze_emb()
            
        hidden_size = self.pretrained_model.config.hidden_size
        output_hidden_dim = args.model_config['output_hidden_dim']
        output_hidden_act_func = args.model_config['output_hidden_act_func']

        self.num_labels = len(args.label_to_id)

        if output_hidden_dim is not None:
            h_dim = [output_hidden_dim, self.num_labels]
        else:
            h_dim = [self.num_labels]

        self.linear = LinearLayer(
            in_dim=hidden_size,
            h_dim=h_dim,
            activation=output_hidden_act_func,
            use_bn=False
        )

        try:
            self.loss_func = nn.CrossEntropyLoss(
                reduction="mean", 
                label_smoothing=args.model_config.get('label_smoothing', 0)
            )
        except Exception as e:
            self.loss_func = nn.CrossEntropyLoss(
                reduction="mean", 
            )
        self.return_logits = False
        self.to(args.device)

    def avg_pool(sel, h, attention_mask):
        attention_mask = attention_mask.unsqueeze(-1)
        h = h.masked_fill(attention_mask==0, float(0))
        h = h.sum(dim=1) / attention_mask.sum(dim=1)        
        return h

    def freeze_emb(self):
        # Freeze all parameters except self attention parameters
        for param_name, param in self.pretrained_model.named_parameters():
            if "embeddings" in param_name:
                param.requires_grad = False

    def set_return_logits(self):
        self.return_logits = True

    def forward(
        self,
        input_ids,
        attention_mask,
        label=None,
    ):
        outputs = dict()
        lm = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            output_attentions=True,
            return_dict=True,
        )
        h = lm["last_hidden_state"]
        h = self.avg_pool(h, attention_mask=attention_mask)
        logits = self.linear(h)

        if self.return_logits:
            return logits 
        else:
            if label is not None:
                loss = self.loss_func(
                    logits.view(-1, self.num_labels), label.view(-1)  # [N, C]  # [N]
                )
            else:
                loss = None
            prediction = torch.argmax(logits, dim=1)
            outputs = NLPModelOutput(
                loss=loss, 
                prediction=prediction, 
                logits=logits
            )
            return outputs
