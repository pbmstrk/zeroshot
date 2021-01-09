import torch
from tqdm import tqdm

class Tester:

    def __init__(self):
        self.model = None
        self.metrics = None

    @staticmethod
    def unzip_batch(batch):
        return list(map(list, zip(*batch)))

    def test_batch(self, batch):

        inputs, targets = self.unzip_batch(batch)
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

