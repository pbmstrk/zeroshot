import os
from collections import OrderedDict

import gensim.downloader
import numpy as np
from tqdm import tqdm

from ..utils import download_extract


def extract_vectors(filepath):

    embedding_map = OrderedDict()
    with open(filepath) as embed_file:
        for line in tqdm(embed_file):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype="float32")
                embedding_map[word] = coefs
            except ValueError:
                continue
    return embedding_map


def GloVe(name, dim, root=".data"):

    r"""
    Load pre-trained GloVe word embeddings. Returns an instance of the Vectors
    class.

    Reference: `GloVe: Global Vectors for Word Representation <https://nlp.stanford.edu/projects/glove/>`_

    Args:
        name: Name of vectors to retrieve - one of 6B, 42B, 840B and twitter.27B
        dim: Dimension of word vectors.
        root: Name of the root directory in which to cache vectors.

    Returns:
        Instance of vectors class.

    Example::

        >>> glove_vectors = GloVe(name="6B", dim=300)
    """

    URLs = {
        "42B": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "840B": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter.27B": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "6B": "https://nlp.stanford.edu/data/glove.6B.zip",
    }

    download_extract(URLs[name], name=name, root=root)
    filename = f"glove.{name}.{dim}d.txt"
    filepath = os.path.join(root, name, filename)
    vector_map = extract_vectors(filepath)

    return Vectors(vector_map)


def Word2Vec():

    r"""
    Load pre-trained Word2Vec embeddings using Gensim.

    Reference: `Word2Vec <https://code.google.com/archive/p/word2vec/>`_

    Example::

        >>> word2vec = Word2Vec()
    """

    return gensim.downloader.load("word2vec-google-news-300")


class Vectors:

    """
    Mimic gensim API - creates a common interface
    """

    def __init__(self, vector_map):
        self.vector_map = vector_map

    @property
    def index2entity(self):
        return list(self.vector_map.keys())

    def __getitem__(self, idx):
        return self.vector_map[idx]
