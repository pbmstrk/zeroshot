import logging

from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)


class PrintingCallback(Callback):
    def __init__(self, monitor):
        if isinstance(monitor, str):
            monitor = [monitor]
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        log.info("Epoch: %d \t Step %d", epoch, trainer.global_step)
        for m in self.monitor:
            log.info("%s: %.4f", m, metrics[m])
