import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from turing.utils import TorchTuple

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainingHeads, PreTrainedBertModel, BertPreTrainingHeads
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

class BertPretrainingLoss(PreTrainedBertModel):
    def __init__(self, bert_encoder, config):
        super(BertPretrainingLoss, self).__init__(config)
        self.bert = bert_encoder
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.cls.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                masked_lm_labels=None,
                next_sentence_label=None):
        sequence_output, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2),
                                          next_sentence_label.view(-1))
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertClassificationLoss(PreTrainedBertModel):
    def __init__(self, bert_encoder, config, num_labels: int = 1):
        super(BertClassificationLoss, self).__init__(config)
        self.bert = bert_encoder
        self.num_labels = num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        scores = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(scores.view(-1, self.num_labels),
                            labels.view(-1, 1))
            return loss
        else:
            return scores


class BertRegressionLoss(PreTrainedBertModel):
    def __init__(self, bert_encoder, config):
        super(BertRegressionLoss, self).__init__(config)
        self.bert = bert_encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        _, pooled_output = self.bert(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1, 1))
            return loss
        else:
            return logits


class BertMultiTask:
    def __init__(self, args):
        self.config = args.config

        if not args.use_pretrain:

            if args.progressive_layer_drop:
                print("BertConfigPreLnLayerDrop")
                from nvidia.modelingpreln_layerdrop import BertForPreTrainingPreLN, BertConfig
            else:
                from nvidia.modelingpreln import BertForPreTrainingPreLN, BertConfig

            bert_config = BertConfig(**self.config["bert_model_config"])
            bert_config.vocab_size = len(args.tokenizer.vocab)

            # Padding for divisibility by 8
            if bert_config.vocab_size % 8 != 0:
                bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
            print("VOCAB SIZE:", bert_config.vocab_size)

            self.network = BertForPreTrainingPreLN(bert_config, args)
        # Use pretrained bert weights
        else:
            self.bert_encoder = BertModel.from_pretrained(
                self.config['bert_model_file'],
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
                'distributed_{}'.format(args.local_rank))
            bert_config = self.bert_encoder.config

        self.device = None

    def set_device(self, device):
        self.device = device

    def save(self, filename: str):
        network = self.network.module
        return torch.save(network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(
            torch.load(model_state_dict,
                       map_location=lambda storage, loc: storage))

    def move_batch(self, batch: TorchTuple, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()
