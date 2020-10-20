import torch
import torch.nn as nn
import os
from pathlib import Path
    

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
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


class SAModel(nn.Module):
    def __init__(self, model_config, num_labels, pretrained_model, device='cpu'):
        super(SAModel, self).__init__()
        self.model_config = model_config
        self.pretrained_model = pretrained_model
        self.num_labels = num_labels

        self.cls_fc_layer = FCLayer(pretrained_model.config.hidden_size, 
                                    pretrained_model.config.hidden_size, 
                                    model_config['classifier']["dropout_rate"])
        self.t_fc_layer = FCLayer(pretrained_model.config.hidden_size, 
                                  pretrained_model.config.hidden_size, 
                                  model_config['classifier']["dropout_rate"])
        
        self.label_classifier = FCLayer(
            pretrained_model.config.hidden_size * 2, 
            num_labels, 
            model_config['classifier']["dropout_rate"], 
            use_activation=False)

        self.device = device
        self.to(device)
        self.pretrained_model.to(device)

    @staticmethod
    def target_average(hidden_output, t_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param t_mask: [batch_size, max_seq_len]
                e.g. t_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        t_mask_unsqueeze = t_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (t_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(t_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, t_mask):
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        outputs = self.pretrained_model.model(input_ids=input_ids, 
                                              attention_mask=attention_mask,
                                              token_type_ids=token_type_ids, 
                                              return_dict=True)['last_hidden_state']

        pooled_output = self.cls_fc_layer(outputs[:, :1, :]).squeeze(1)

        # Average
        t_h = self.target_average(outputs, t_mask)
        t_h = self.t_fc_layer(t_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, t_h], dim=-1)
        logits = self.label_classifier(concat_h)

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits  
    
    def save_state(self, model_dir, suffix=""):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = Path(model_dir)
        if suffix:
            state_path = model_dir / f"state_dict_{suffix}.pt"
        else:
            state_path = model_dir / f"state_dict.pt"
        torch.save(self.state_dict(), state_path)

    def load_state(self, model_dir):
        assert(os.path.exists(model_dir))
        model_dir = Path(model_dir)
        for fnm in os.listdir(model_dir):
            if fnm.startswith("state_dict") and fnm.endswith(".pt"):
                self.load_state_dict(torch.load(model_dir / fnm))  
                break 
