import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from model.utils import load_pretrained_bert, load_pretrained_config
from model.layer.fc import FCLayer


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


class TDBERT(BertPreTrainedModel):
    INPUT = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "target_mask",
        "label",
    ]
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
            if "embeddings" in param_name:
                param.requires_grad = False

    def forward(
        self,
        input_ids,
        target_mask,
        attention_mask,
        token_type_ids,
        label=None,
        **kwargs
    ):
        lm = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        h = lm["last_hidden_state"]
        # Average or max
        tgt_h = self.pool_target(
            h, target_mask
        )  # outputs: [B, S, Dim], target_mask: [B, S]

        tgt_h = self.fc_layer(tgt_h)
        logits = self.classifier(tgt_h)
        prediction = torch.argmax(logits, dim=1).cpu().tolist()

        if label is not None:
            loss = self.loss_func(logits.view(-1, self.num_labels), label.view(-1))
            loss = loss.mean()
        else:
            loss = None
        return [loss, prediction, logits]
