import torch
import torch.nn as nn

from nlp_pipeline.model.layer.fc import LinearLayer
from nlp_pipeline.model.layer.cnn import ConvLayer
from nlp_pipeline.model.layer.embedding import WordEmbeddings
from nlp_pipeline.model.utils import NLPModelOutput


class TEXT_CNN(nn.Module):
    def __init__(self, args):
        super(TEXT_CNN, self).__init__()
        # hyper params
        kernel_size = args.model_config["kernel_size"]  # single size
        kernel_num = args.model_config["kernel_num"]
        cnn_dropout = args.model_config["cnn_dropout"]
        cnn_use_bn = args.model_config["cnn_use_bn"]
        pool_method = args.model_config["pool_method"]
        output_hidden_dim = args.model_config["output_hidden_dim"]
        output_hidden_act_func = args.model_config["output_hidden_act_func"]
        output_use_bn = args.model_config["output_use_bn"]

        if type(kernel_size) == int:
            kernel_size = [kernel_size]

        self.emb = WordEmbeddings(
            pretrained_emb_path=args.pretrained_emb_path,
            embedding_trainable=args.model_config["embedding_trainable"],
            emb_dim=args.model_config["emb_dim"],
            vocab_size=args.vocab_size,
            emb_dropout=args.model_config["emb_dropout"],
            word_to_id=args.word_to_id
        )
        emb_dim = self.emb.emb_dim
        self.num_labels = len(args.label_to_id)
        self.conv = ConvLayer(
            emb_dim, kernel_num, kernel_size, pool_method=pool_method, use_bn=cnn_use_bn
        )
        self.cnn_dp = nn.Dropout(cnn_dropout) if cnn_dropout > 0.0 else None

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

    def forward(self, input_ids, attention_mask, label=None, **kwargs):
        outputs = dict()
        x = self.emb(input_ids)  # [B, L, E]
        x = self.conv(x, attention_mask)  # [B, Kn*#ks]

        if self.cnn_dp is not None:
            x = self.cnn_dp(x)
        logits = self.linear(x)  # [B, Nc]

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