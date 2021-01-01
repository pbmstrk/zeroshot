import torch
from transformers import AutoTokenizer
from transformers import AutoModel
from tqdm import tqdm


def get_projection_matrix(model_name, vectors, k=1000):

    words = list(vectors.keys())[:k]
    dim = vectors[words[0]].shape[0]

    # get wordvec matrix
    print("Obtaining word vector matrix")
    wordvec_matrix = torch.zeros((k, dim))
    for i, word in tqdm(enumerate(words)):
        wordvec_matrix[i] = torch.tensor(vectors[word])

    print("Obtaining encoder matrix")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    batch_size = 32
    encoder_matrix = torch.Tensor()
    with torch.no_grad():
        for i in tqdm(range(0, len(words), batch_size)):
            batch = words[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding="longest")
            outputs = model(**inputs)[0]
            outputs = outputs.mean(1)
            encoder_matrix = torch.cat((encoder_matrix, outputs))

    print("Obtaining least squares estimate")
    return torch.lstsq(wordvec_matrix, encoder_matrix).solution