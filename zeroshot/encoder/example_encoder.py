import torch


class ExampleEncoder:

    def __init__(self, tokenizer, target_encoding):
        self.tokenizer = tokenizer
        self.target_encoding = target_encoding

    def __call__(self, text, target=None, **kwargs):

        if isinstance(text, str):
            text = [text]
        if isinstance(target, str):
            target = [target]

        encoded_text = self.tokenizer(text, **kwargs)

        if target:
            encoded_targets = torch.tensor(list(map(self.target_encoding.get, target)))
            return encoded_text, encoded_targets

        return encoded_text

    def collate_fn(self, batch):

        texts, targets = self.unzip_batch(batch)

        return self(text=texts, target=targets, return_tensors='pt', padding='longest',
                truncation=True)

    def unzip_batch(self, batch):
        return list(map(list, zip(*batch)))

