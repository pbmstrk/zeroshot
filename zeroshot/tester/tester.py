import torch
from tqdm import tqdm

class Tester:

    def __init__(self, target_encoding):
        self.model = None
        self.metrics = None
        self.target_encoding = target_encoding

    @staticmethod
    def unzip_batch(batch):
        return list(map(list, zip(*batch)))

    def test_batch(self, batch):

        inputs, targets = self.unzip_batch(batch)
        targets = torch.tensor(list(map(self.target_encoding.get, targets)))
        outputs = self.model(**inputs)
        
        _, pred = torch.max(outputs.data, 1)
        batch_correct = (pred.detach().cpu() == targets).sum()
        self.metrics["acc"] = self.metrics.get("acc", 0) + batch_correct

    def test(self, model, dataloader):
        
        self.model = model
        self.metrics = {}

        self.model.eval()
        for batch in tqdm(dataloader):
            self.test_batch(batch)

        return self.metrics

