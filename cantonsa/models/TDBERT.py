import torch
import torch.nn as nn
from transformers import AlbertModel, BertModel, BertPreTrainedModel, RobertaModel
from cantonsa.models.base import BaseModel


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
    INPUT_COLS = ["raw_text", "attention_mask", "token_type_ids", "target_mask", "label"]
    def __init__(
        self,
        model_config,
        num_labels,
        pretrained_emb=None,
        num_emb=None,
        pretrained_lm=None,
        target_pooling="max",
        device="cpu",
    ):
        super(TDBERT, self).__init__(pretrained_lm.config)
        assert target_pooling in ["mean", "max"]
        self._device = device
        self.model_config = model_config
        self.pretrained_lm = pretrained_lm.model
        self.pretrained_lm_config = pretrained_lm.config
        self.num_labels = num_labels
        self.target_pooling = target_pooling
        self.init_classifier()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.to(self._device)

    def init_classifier(self):
        """
        TODO: (1) decide the activations, (2) chain it into one object
        """
        
        self.tgt_fc_layer = FCLayer(
            input_dim=self.pretrained_lm_config.hidden_size,
            output_dim=self.pretrained_lm_config.hidden_size,
            dropout_rate=self.model_config["dropout_rate"],
        )
        
        if self.model_config.get("use_cls", False):
            self.cls_fc_layer = FCLayer(
                input_dim=self.pretrained_lm_config.hidden_size,
                output_dim=self.pretrained_lm_config.hidden_size,
                dropout_rate=self.model_config["dropout_rate"],
            )
            label_classifier_input_dim = self.pretrained_lm_config.hidden_size * 2
        else:
            label_classifier_input_dim = self.pretrained_lm_config.hidden_size
             
        self.label_classifier = FCLayer(
            input_dim=label_classifier_input_dim,
            output_dim=self.num_labels,
            dropout_rate=self.model_config["dropout_rate"],
            use_activation=False,
        )
        self.to(self._device)

    def pool_target(self, hidden_output, t_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param t_mask: [batch_size, max_seq_len]
                e.g. t_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        if self.target_pooling == "mean":
            t_mask_unsqueeze = t_mask.unsqueeze(1)  # [b, 1, j-i+1]
            length_tensor = (t_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
            sum_vector = torch.bmm(t_mask_unsqueeze.float(), hidden_output).squeeze(
                1
            )  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
            avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
            return avg_vector
        elif self.target_pooling == "max":
            t_h = torch.max(
                hidden_output * torch.unsqueeze(t_mask, -1), dim=1, keepdim=False
            )
            max_vector = t_h.values
            return max_vector
        else:
            raise (Exception)

    def freeze_lm(self):
        # Freeze all parameters except self attention parameters
        for param_name, param in self.pretrained_lm.named_parameters():
            if "selfatt" not in param_name and "fc" not in param_name:
                param.requires_grad = False

    def unfreeze_lm(self):
        # Unfreeze all parameters except self attention parameters
        for param_name, param in self.pretrained_lm.named_parameters():
            if "selfatt" not in param_name and "fc" not in param_name:
                param.requires_grad = True

    def forward(
        self,
        raw_text,
        target_mask,
        attention_mask,
        token_type_ids,
        label=None,
        return_reps=False
    ):
        outputs = self.pretrained_lm(
            input_ids=raw_text,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )["last_hidden_state"]

        # Average
        tgt_h = self.pool_target(
            outputs, target_mask
        )  # outputs: [B, S, Dim], target_mask: [B, S]

        tgt_h = self.tgt_fc_layer(tgt_h)

        if self.model_config.get("use_cls", False):
            cls_h = self.cls_fc_layer(outputs[:, :1, :]).squeeze(1)
            # Concat -> fc_layer
            h = torch.cat([cls_h, tgt_h], dim=-1)
        else:
            h = tgt_h

        logits = self.label_classifier(h)

        if label is not None:
            losses = self.loss_func(logits.view(-1, self.num_labels), label.view(-1))

        if return_reps:
            return losses, logits, tgt_h
        else:
            return losses, logits
