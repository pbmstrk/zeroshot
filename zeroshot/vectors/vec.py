import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import gensim.downloader

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

def GloVe(name, dim, root = ".data"):

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
    return gensim.downloader.load('word2vec-google-news-300')


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

    