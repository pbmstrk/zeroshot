import pytorch_lightning as pl
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

    def forward(self, text_encodings, label_encodings):
        return torch.einsum('ij, kj->ik', text_encodings, label_encodings)


class BiEncoderClassifier(pl.LightningModule):

    def __init__(self, text_encoder, label_encodings, optimizer,
            scheduler = None):
        super().__init__()
        self.encoder = text_encoder
        self.scorer = Scorer()
        self.label_encodings = label_encodings
        self.opt = optimizer
        if scheduler is not None:
            self.schedule = scheduler

    def forward(self, input_ids, **kwargs):
        text_encodings = self.encoder(input_ids, **kwargs)[1]
        return self.scorer(text_encodings, self.label_encodings)

    def step(self, batch, batch_idx, prefix=""):

        inputs, targets = batch

        outputs = self(**inputs)

        # compute loss
        loss = F.cross_entropy(outputs, targets)

        # compute acc
        _, pred = torch.max(outputs.data, 1)
        correct = (pred == targets).sum()
        acc = correct.float() / len(targets)

        self.log(prefix + "loss", loss)
        self.log(prefix + "acc", acc, on_epoch=True, on_step=False)

        return loss

    def training_step(self, batch, batch_idx):

        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):

        return self.step(batch, batch_idx, prefix="val_")

    def test_step(self, batch, batch_idx):

        return self.step(batch, batch_idx, prefix="test_")

    def configure_optimizers(self):

        if hasattr(self, "schedule"):
            return [self.opt], [self.schedule]

        return self.opt

