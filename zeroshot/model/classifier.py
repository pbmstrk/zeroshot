import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)


class LabelEncoder(nn.Module):

    def __init__(self, model_name, no_grad=True):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.no_grad = no_grad

    def forward(self, input_ids, **kwargs):
        if self.no_grad:
            with torch.no_grad():
                return self.model(input_ids, **kwargs)
        return self.model(input_ids, **kwargs)


class Scorer(nn.Module):

    def __init__(self):
        super().__init__()
        self.score = nn.CosineSimilarity()

    def forward(self, text_encodings, label_encodings):
        return torch.stack([self.score(tensor, label_encodings) for tensor in text_encodings]) 


class ZeroShotClassifier(nn.Module):

    def __init__(self, text_encoder, projection_matrix, label_encodings):
        super().__init__()
        self.encoder = text_encoder
        self.scorer = Scorer()
        self.register_buffer("label_encodings", label_encodings)
        self.register_buffer("projection_matrix", projection_matrix)

    def forward(self, input_ids, **kwargs):
        text_encodings = self.encoder(input_ids, **kwargs)[1]
        proj_text_enc = text_encodings @ self.projection_matrix
        proj_label_enc = self.label_encodings @ self.projection_matrix
        return self.scorer(proj_text_enc, proj_label_enc)