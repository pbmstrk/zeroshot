import torch
from transformers import AutoTokenizer
from transformers import AutoModel


def get_projection_matrix(model_name, vectors, k=1000):

    words = list(vectors.keys())[:k]
    dim = vectors[words[0]].shape[1]

    # get wordvec matrix
    print("Obtaining word vector matrix")
    wordvec_matrix = torch.zeros((k, dim))
    for i, word in enumerate(words):
        wordvec_matrix[i] = vectors[word]

    print("Obtaining encoder matrix")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    batch_size = 32
    encoder_matrix = torch.tensor([])
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding="longest")
        outputs = model(**inputs)[1]
        encoder_matrix = torch.stack((encoder_matrix, outputs))

    print("Obtaining least squares estimate")
    return torch.lstsq(wordvec_matrix, encoder_matrix)

    

