import torch
import torch.nn as nn

from ..utils import move_args_to_device
from ..model import ZeroShotTopicClassifier


class ZeroShotPipeline(nn.Module):

    r"""
    ZeroShot Classifier.

    Args:
        model_name: Name of model to use.

    Example::
        
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-based-uncased')
        >>> model = AutoModel.from_pretrained('deepset/sentence_transformers')
        >>> pipeline = ZeroShotPipeline(tokenizer, model)
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device("cpu")

    def __call__(self, inputs, tokenizer_options={}):

        """
        call model
        """

        tokenizer_options = self._add_tokenizer_defaults(tokenizer_options)
        encoded_inputs = self.tokenizer(inputs, **tokenizer_options)

        return self.forward(**encoded_inputs)

    @move_args_to_device
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def add_labels(self, labels, tokenizer_options={}):
        """
        add labels
        """

        tokenizer_options = self._add_tokenizer_defaults(tokenizer_options)
        encoded_labels = self.tokenizer(labels, **tokenizer_options)
        self.model.create_label_index(**encoded_labels)

    def add_projection_matrix(self, proj_mat):

        """
        add projection matrix
        """
        self.model.projection_matrix = proj_mat

    @staticmethod
    def _add_tokenizer_defaults(options):
        options.setdefault("return_tensors", "pt")
        options.setdefault("padding", "longest")
        options.setdefault("truncation", True)
        options.setdefault("max_length", 512)
        return options

    def to(self, device):
        self.model.to(device)
        self.device = device