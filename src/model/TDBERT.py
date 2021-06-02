import torch
import torch.nn as nn
from transformers import AlbertModel, BertModel, BertPreTrainedModel, RobertaModel
from .model_utils import BaseModel, load_pretrained_bert, load_pretrained_config


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class TDBERT(BertPreTrainedModel, BaseModel):
    INPUT = [
        "raw_text",
        "attention_mask",
        "token_type_ids",
        "target_mask",
        "label",
    ]
    MODEL_TYPE = "bert"
    def __init__(self, args):
        pretrained_config = load_pretrained_config(
            args.model_config['pretrained_lm']
        )
        super(TDBERT, self).__init__(pretrained_config)
        
        # assert target_pooling in ["mean", "max"]
        self.model_config = args.model_config
        self.pretrained_model = load_pretrained_bert(
            self.model_config['pretrained_lm']
        )
        if not args.model_config["embedding_trainable"]:
            self.freeze_emb()

        self.pretrained_config = pretrained_config
        self.num_labels = len(args.label_to_id)
        self.init_classifier()
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        self.to(args.device)

    def init_classifier(self):
        self.fc_layer = FCLayer(
            input_dim=self.pretrained_config.hidden_size,
            output_dim=self.pretrained_config.hidden_size,
            dropout_rate=self.model_config["dropout_rate"],
        )
        self.classifier = FCLayer(
            input_dim=self.pretrained_config.hidden_size,
            output_dim=self.num_labels,
            dropout_rate=self.model_config["dropout_rate"],
            use_activation=False,
        )

    def pool_target(self, hidden_output, t_mask):
        """Pool the entity hidden state vectors (H_i ~ H_j)
        """
        t_h = torch.max(
            hidden_output.float() * torch.unsqueeze(t_mask.float(), -1),
            dim=1,
            keepdim=False,
        )
        return t_h.values

    def freeze_emb(self):
        # Freeze all parameters except self attention parameters
        for param_name, param in self.pretrained_model.named_parameters():
            if "selfatt" not in param_name and "fc" not in param_name:
                param.requires_grad = False

    def unfreeze_emb(self):
        # Unfreeze all parameters except self attention parameters
        for param_name, param in self.pretrained_model.named_parameters():
            if "selfatt" not in param_name and "fc" not in param_name:
                param.requires_grad = True

    def forward(
        self,
        raw_text,
        target_mask,
        attention_mask,
        token_type_ids,
        label=None,
        return_tgt_pool=False,
        return_tgt_mask=False,
        return_all_repr=False,
        return_attn=False,
        **kwargs
    ):
        lm = self.pretrained_model(
            input_ids=raw_text,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_attentions=return_attn,
        )

        h = lm["last_hidden_state"]
        # Average or max
        tgt_h = self.pool_target(
            h, target_mask
        )  # outputs: [B, S, Dim], target_mask: [B, S]

        tgt_h = self.fc_layer(tgt_h)
        logits = self.classifier(tgt_h)

        if label is not None:
            losses = self.loss_func(logits.view(-1, self.num_labels), label.view(-1))
        else:
            losses = None

        outputs = [losses.mean(), logits]

        if return_tgt_pool:
            outputs += [tgt_h]
        else:
            outputs += [None]

        if return_tgt_mask:
            outputs += [target_mask]
        else:
            outputs += [None]

        if return_all_repr:
            outputs += [h]
        else:
            outputs += [None]

        if return_attn:
            outputs += [lm["attentions"]]
        else:
            outputs += [None]

        return outputs
