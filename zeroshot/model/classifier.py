import torch
import torch.nn as nn
from transformers import BertModel

from ..utils import MisconfigurationError


class TextEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, **kwargs):
        mask = attention_mask
        outputs = self.model(input_ids, attention_mask, **kwargs)[0]
        mean_pool = (outputs*mask.unsqueeze(-1)).sum(1) / (mask.unsqueeze(-1).sum(1))
        return mean_pool


class Scorer(nn.Module):

    def __init__(self):
        super().__init__()
        self.score = nn.CosineSimilarity()

    def forward(self, text_encodings, label_encodings):
        return torch.stack([self.score(tensor.unsqueeze(0), label_encodings) for tensor in text_encodings]) 


class ZeroShotClassifier(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.encoder = TextEncoder(model_name)
        self.scorer = Scorer()
        self._use_projection_matrix = False
        self.register_buffer("label_encodings", torch.Tensor())
        self.register_buffer("proj_mat", torch.Tensor())

    def forward(self, input_ids, **kwargs):
        text_encodings = self.encoder(input_ids, **kwargs)
        label_encodings = self.label_encodings
        
        if self.use_projection_matrix:
            text_encodings = text_encodings @ self.proj_mat
            if self.label_encodings.nelement():
                label_encodings = label_encodings @ self.proj_mat
        
        if self.label_encodings.nelement():
            return self.scorer(text_encodings, label_encodings)
        
        return text_encodings

    def add_projection_matrix(self, proj_mat):
        self.register_buffer("proj_mat", proj_mat)

    @property
    def use_projection_matrix(self):
        return self._use_projection_matrix

    @use_projection_matrix.setter
    def use_projection_matrix(self, value):
        if not isinstance(value, bool):
            raise ValueError("Value must be a boolean")
        if not self.proj_mat.nelement() and value:
            raise MisconfigurationError("Trying to use projection matrix, but none found. Please add matrix using add_projection_matrix()")
        self._use_projection_matrix = value

    def add_labels(self, input_ids, **kwargs):
        cur_device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(cur_device)
        kwargs = {
            key: value.to(cur_device) if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        with torch.no_grad():
            new_label_encodings = self.encoder(input_ids, **kwargs)
        self.label_encodings = torch.cat((self.label_encodings, new_label_encodings))

    def reset_labels(self):
        cur_device = next(self.encoder.parameters()).device
        self.register_buffer("label_encodings", torch.Tensor().to(cur_device))


