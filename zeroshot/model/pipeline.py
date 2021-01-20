import torch

from ..tokenizer import ZeroShotTopicTokenizer
from ..utils import move_args_to_device
from .classifier import ZeroShotTopicClassifier


class ZeroShotPipeline:

    r"""
    ZeroShot Classifier.

    Args:
        model_name: Name of model to use.

    Example::

        >>> pipeline = ZeroShotPipeline(model_name="deepset/sentence_bert")
    """

    def __init__(self, model_name):
        self.classifier = ZeroShotTopicClassifier(model_name)
        self.tokenizer = ZeroShotTopicTokenizer(model_name)
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
        return self.classifier(**kwargs)

    def add_labels(self, labels, tokenizer_options={}):
        """
        add labels
        """

        tokenizer_options = self._add_tokenizer_defaults(tokenizer_options)
        encoded_labels = self.tokenizer(labels, **tokenizer_options)
        self.classifier.create_label_index(**encoded_labels)

    def add_projection_matrix(self, proj_mat):

        """
        add projection matrix
        """
        self.classifier.projection_matrix = proj_mat

    @staticmethod
    def _add_tokenizer_defaults(options):
        options.setdefault("return_tensors", "pt")
        options.setdefault("padding", "longest")
        options.setdefault("truncation", True)
        options.setdefault("max_length", 512)
        return options

    def to(self, device):
        self.classifier.to(device)
        self.device = device

    def eval(self):
        self.classifier.eval()
