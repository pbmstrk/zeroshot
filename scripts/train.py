import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer

from zeroshot.model import LabelEncoder, TextEncoder, BiEncoderClassifier
from zeroshot.data import ZeroShotTopicClassificationDataset
from zeroshot.encoder import ExampleEncoder
from zeroshot.datamodule import DataModule
from zeroshot.callbacks import PrintingCallback

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):

    log.info("Arguments:\n %s", OmegaConf.to_yaml(cfg))

    seed_everything(cfg.random.seed)

    description_path = hydra.utils.to_absolute_path(cfg.dataset.descriptions)

    with open(description_path, "r") as f:
        descriptions = [ln.strip() for ln in f.readlines()]
    
    # get dataset
    train, val, test, classes_dict = ZeroShotTopicClassificationDataset(return_subset=cfg.dataset.subset) 

    # get tokenizer and example encoder - convert string to ids
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    encoder = ExampleEncoder(tokenizer, target_encoding=classes_dict)
    dm = DataModule(train=train, collate_fn=encoder.collate_fn, val=val, test=test, batch_size=cfg.datamodule.batch_size)

    text_encoder = TextEncoder(model_name=cfg.model.text_encoder.name)
    label_encoder = LabelEncoder(model_name=cfg.model.label_encoder.name)
    
    description_encodings = tokenizer(descriptions, padding="longest", return_tensors="pt")
    label_encodings = label_encoder(**description_encodings)[1]

    optimizer = torch.optim.Adam(text_encoder.parameters(), **cfg.optimizer.args)

    model = BiEncoderClassifier(
        text_encoder=text_encoder,
        label_encodings=label_encodings,
        optimizer=optimizer
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=hydra.utils.to_absolute_path(cfg.checkpoint_path),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(checkpoint_callback=checkpoint_callback,
        callbacks=[PrintingCallback(["val_loss", "val_acc"]), early_stop_callback], **cfg.trainer)

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())


if __name__ == "__main__":
    main()