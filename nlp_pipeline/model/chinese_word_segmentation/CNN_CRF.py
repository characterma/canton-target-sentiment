import torch.nn as nn

from nlp_pipeline.model.layer.crf import LinearChainCRF
from nlp_pipeline.model.layer.cnn import ConvLayer
from nlp_pipeline.model.layer.embedding import WordEmbeddings
from nlp_pipeline.model.utils import NLPModelOutput


class CNN_CRF(nn.Module):
    def __init__(self, args):
        super(CNN_CRF, self).__init__()

        # hyper params
        kernel_size = args.model_config["kernel_size"]  # single size
        kernel_num = args.model_config["kernel_num"]
        cnn_dropout = args.model_config["cnn_dropout"]
        cnn_use_bn = args.model_config["cnn_use_bn"]

        if type(kernel_size) == int:
            kernel_size = [kernel_size]
        fc_in = kernel_num * len(kernel_size)

        self.embed = WordEmbeddings(
            pretrained_emb_path=args.pretrained_emb_path,
            embedding_trainable=args.model_config["embedding_trainable"],
            emb_dim=args.model_config["emb_dim"],
            vocab_size=args.vocab_size,
            emb_dropout=args.model_config["emb_dropout"],
            word_to_id=args.word_to_id
        )
        emb_dim = self.embed.emb_dim

        # model init
        self.cnns = nn.ModuleList(
            [
                ConvLayer(
                    emb_dim,
                    kernel_num,
                    kernel_size,
                    pool_method=None,
                    use_bn=cnn_use_bn,
                    keep_seq_length=True,
                )
                for i in range(args.model_config["cnn_layers"])
            ]
        )
        self.cnn_dp = nn.Dropout(cnn_dropout) if cnn_dropout > 0.0 else None
        self.crf = LinearChainCRF(fc_in, len(args.label_to_id))
        self.return_logits = False
        self.to(args.device)

    def set_return_logits(self):
        self.return_logits = True

    def forward(self, input_ids, attention_mask,  label=None, **kwargs):
        outputs = dict()
        x = self.embed(input_ids)

        for cnn in self.cnns:
            x = cnn(x, length_index=attention_mask)

        if self.cnn_dp is not None:
            logits = self.cnn_dp(x)
        else:
            logits = x


        if self.return_logits:
            return logits 
        else:
            prediction, scores = self.crf.viterbi_decode(
                logits, length_index=attention_mask, top_k=1
            )  # [B, k, L], [B,K]
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