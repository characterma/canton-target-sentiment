import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_pipeline.model.layer.embedding import WordEmbeddings
from nlp_pipeline.model.layer.fc import LinearLayer
from nlp_pipeline.model.utils import NLPModelOutput


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        kernel_size = args.model_config["kernel_size"]
        vocab_size = args.vocab_size
        emb_dim=args.model_config["emb_dim"]
        kernel_num = args.model_config["kernel_num"]
        cnn_dropout = args.model_config["cnn_dropout"]
        output_hidden_dim = args.model_config["output_hidden_dim"]
        output_hidden_act_func = args.model_config["output_hidden_act_func"]
        output_use_bn = args.model_config["output_use_bn"]

        self.max_length = args.model_config["max_length"]
        if type(kernel_size) == int:
            kernel_size = [kernel_size]
            
        self.emb = WordEmbeddings(
            pretrained_emb_path=args.pretrained_emb_path,
            embedding_trainable=args.model_config["embedding_trainable"],
            emb_dim=emb_dim,
            vocab_size=vocab_size,
            emb_dropout=args.model_config["emb_dropout"],
            word_to_id=args.word_to_id
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, emb_dim)) for k in kernel_size]
        )
        self.num_labels = len(args.label_to_id)
        self.dropout = nn.Dropout(cnn_dropout)

        fc_in = kernel_num * len(kernel_size)
        self.linear = LinearLayer(
            in_dim=fc_in,
            h_dim=[output_hidden_dim, self.num_labels],
            activation=output_hidden_act_func,
            use_bn=output_use_bn,
        )

        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.return_logits = False
        self.to(args.device)


    def set_return_logits(self):
        self.return_logits = True


    def conv_block(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, self.max_length - 2).squeeze(2)
        return x
    
    def forward(self, input_ids, label=None, **kwargs):
        out = self.emb(input_ids).unsqueeze(1)

        out = torch.cat([self.conv_block(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        logits = self.linear(out)

        if self.return_logits:
            return logits 
        else:

            prediction = torch.argmax(logits, dim=1)
            if label is not None:
                loss = self.loss_func(
                    logits.view(-1, self.num_labels), label.view(-1)  # [N, C]  # [N]
                )
            else:
                loss = None

            outputs = NLPModelOutput(
                loss=loss, 
                prediction=prediction, 
                logits=logits
            )

            return outputs