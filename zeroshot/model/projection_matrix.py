import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_projection_matrix(model_name, vectors, k=1000, lmbda=0):

    r"""
    Function for calculating a projection matrix from sentence representations to word vectors.

    Args:
        model_name: Model used to obtain sentence representations
        vectors: Object containing the word vectors. 
        k: Number of words to use when calculating least squares projection matrix.
        lmbda: Regularisation parameter.
    """

    words = vectors.index2entity[:k]
    wordvec_matrix = get_wordvec_matrix(words, vectors, k)

    encoder_matrix = get_encoder_matrix(words, model_name)

    return regularized_lstsq(encoder_matrix, wordvec_matrix, lmbda)[:encoder_matrix.shape[1]]


def get_wordvec_matrix(words, vectors, k):

    dim = vectors[words[0]].shape[0]

    wordvec_matrix = torch.zeros((k, dim))
    for i, word in tqdm(enumerate(words)):
        wordvec_matrix[i] = torch.tensor(vectors[word])

    return wordvec_matrix


def get_encoder_matrix(words, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    batch_size = 32
    encoder_matrix = torch.Tensor()
    with torch.no_grad():
        for i in tqdm(range(0, len(words), batch_size)):
            batch = words[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding="longest")
            outputs = model(**inputs)[0].mean(1)
            encoder_matrix = torch.cat((encoder_matrix, outputs))

    return encoder_matrix


def regularized_lstsq(A, B, lmbda):
    n_col = A.shape[1]
    return torch.lstsq(A.T @ B, A.T @ A + lmbda * torch.eye(n_col)).solution
