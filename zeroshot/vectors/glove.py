import os
from collections import OrderedDict

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

def GloVe(name, root = ".data"):


    URLs = {
        "42B": "https://nlp.stanford.edu/data/glove.42B.300d.zip",
        "840B": "https://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter.27B": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "6B": "https://nlp.stanford.edu/data/glove.6B.zip",
    }

    filepath = download_extract(URLs[name], name=name, root=root)
    vector_map = extract_vectors(filepath)

    return vector_map