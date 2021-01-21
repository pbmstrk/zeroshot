import torch
import torch.nn as nn
from transformers import BertModel


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, **kwargs):
        mask = attention_mask
        outputs = self.model(input_ids, attention_mask, **kwargs)[0]
        mean_pool = (outputs * mask.unsqueeze(-1)).sum(1) / (mask.unsqueeze(-1).sum(1))
        return mean_pool


class Scorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.CosineSimilarity()

    def forward(self, text_encodings, label_encodings):
        return torch.stack(
            [
                self.score(tensor.unsqueeze(0), label_encodings)
                for tensor in text_encodings
            ]
        )


class ZeroShotTopicClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = TextEncoder(model)
        self.scorer = Scorer()
        self._use_projection_matrix = False
        self.register_buffer("label_encodings", torch.Tensor())
        self.register_buffer("proj_mat", torch.Tensor())

    def forward(self, input_ids, **kwargs):
        text_encodings = self.model(input_ids, **kwargs)
        assert self.label_encodings.nelement()
        label_encodings = self.label_encodings

        if self._use_projection_matrix:
            text_encodings = text_encodings @ self.proj_mat
            label_encodings = label_encodings @ self.proj_mat

        scores = self.scorer(text_encodings, label_encodings)
        return scores

    @property
    def projection_matrix(self):
        if self._use_projection_matrix:
            return self.proj_mat
        return None

    @projection_matrix.setter
    def projection_matrix(self, value):
        if not isinstance(value, torch.Tensor):
            if not value is None:
                raise ValueError("Value must be a tensor or None")
        if value is None:
            self._use_projection_matrix = False
        else:
            self.register_buffer("proj_mat", value)
            self._use_projection_matrix = True

    def create_label_index(self, input_ids, **kwargs):
        self.register_buffer("label_encodings", self.model(input_ids, **kwargs))
