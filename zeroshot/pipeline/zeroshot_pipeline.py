import torch
import torch.nn as nn
from typing import Union, List, Dict, Optional

from ..utils import move_args_to_device
from ..model import ZeroShotTopicClassifier


class ZeroShotPipeline:

    r"""
    Pipeline for zero-shot classification. Uses semantic similarity to predict the most likely label for an input. 

    Args:
        tokenizer: Tokenizer to convert strings into inputs ready for model
        model: Classifier to use for prediction.


    The pipeline is designed to accept tokenizers from Huggingface. In theory any tokenizer is supported, with the requirements that the tokenizer returns a dict which can be used as input to the model.

    Example::
        
        >>> tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
        >>> model = AutoModel.from_pretrained('deepset/sentence_bert')
        >>> pipeline = ZeroShotPipeline(tokenizer, model)
    """

    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = ZeroShotTopicClassifier(model)
        self.device = torch.device("cpu")

    def __call__(self, inputs: Union[str, List[str]], tokenizer_options: Optional[Dict] = None):

        """
        Predict labels for inputs. Labels must be added to the model before making predictions.

        Args:
            inputs: sequence(s) to classify
            tokenizer_options: optional parameters that can be passed to the tokenizer. If None the tokenizer defaults to truncation and padding with a max length of 512.
        """

        tokenizer_options = self._add_tokenizer_defaults(tokenizer_options)
        encoded_inputs = self.tokenizer(inputs, **tokenizer_options)

        return self.forward(**encoded_inputs)

    @move_args_to_device
    def forward(self, **kwargs):
        return self.model(**kwargs)

    def add_labels(self, labels: Union[str, List[str]], tokenizer_options: Optional[Dict] = None):
        """
        Add labels to pipeline. This ensures that predictions for sequences can be made. 

        Args:
            labels: labels on which to make predictions
            tokenizer_options: parameters to pass to the tokenizer to influence behaviour.
        """

        tokenizer_options = self._add_tokenizer_defaults(tokenizer_options)
        encoded_labels = self.tokenizer(labels, **tokenizer_options)
        self.model.create_label_index(**encoded_labels)

    def add_projection_matrix(self, proj_mat: torch.Tensor):

        """
        Add projection matrix to pipeline. The projection matrix is applied to both input encoding and label encoding before computing the similarity.

        Args:
            proj_mat: projection matrix to use during prediction.
        """

        self.model.projection_matrix = proj_mat

    @staticmethod
    def _add_tokenizer_defaults(options):
        if options is None:
            options = {}
        options.setdefault("return_tensors", "pt")
        options.setdefault("padding", "longest")
        options.setdefault("truncation", True)
        options.setdefault("max_length", 512)
        return options

    def to(self, device):
        self.model.to(device)
        self.device = device