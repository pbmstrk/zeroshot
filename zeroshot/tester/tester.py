import torch
from tqdm import tqdm

from ..utils import move_args_to_device


class Tester:

    def __init__(self, gpu=True):
        self.gpu = gpu
        self.device = self.set_device()

    def test_batch(self, batch):
        inputs, targets = batch
        outputs = self.forward(**inputs)
        
        _, pred = torch.max(outputs.data, 1)
        batch_correct = (pred.detach().cpu() == targets).sum()
        self.metrics["acc"] = self.metrics.get("acc", 0) + batch_correct

    def test(self, model, dataloader):
        
        self.model = model
        self.metrics = {}
        self.model.to(self.device)

        for batch in tqdm(dataloader):
            self.test_batch(batch)

        return self.metrics

    def set_device(self):
        return torch.device("cuda") if self.gpu else torch.device("cpu")

    @move_args_to_device
    def forward(self, input_ids, **kwargs):
        with torch.no_grad():
            return self.model(input_ids, **kwargs)

