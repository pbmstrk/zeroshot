import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, x, attention_mask, **kwargs):
        return self.model(x, attention_mask, **kwargs)


class LabelEncoder(nn.Module):

    def __init__(self, model_name, no_grad=True):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.no_grad = no_grad

    def forward(self, x, attention_mask, **kwargs):
        if self.no_grad:
            with torch.no_grad():
                return self.model(x, attention_mask, **kwargs)
        return self.model(x, attention_mask, **kwargs)


class BiEncoderClassifier(pl.LightningModule):

    def __init__(self, text_encoder, label_encodings):
        super().__init__()
        self.encoder = text_encoder
        self.label_encodings = label_encodings

    def forward(self, x):
        pass
