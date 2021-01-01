from torch.utils.data import DataLoader


class DataModule:
    def __init__(self, train=None, collate_fn=None, val=None, test=None, batch_size=16):
        
        
        self.train = train
        self.val = val
        self.test = test
        self.collate_fn = collate_fn
        self.batch_size = batch_size

    def train_dataloader(self):

        if not self.train:
            raise ValueError

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):

        if self.val:
            raise ValueError

        return DataLoader(
            self.val, batch_size=self.batch_size, collate_fn=self.collate_fn
        )

    def test_dataloader(self):

        if self.test:
            raise ValueError

        return DataLoader(
            self.test, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
