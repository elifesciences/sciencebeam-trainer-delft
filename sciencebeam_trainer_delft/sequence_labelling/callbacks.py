import logging
import warnings

import numpy as np

import keras.backend as K
from keras.callbacks import Callback

from sciencebeam_trainer_delft.sequence_labelling.saving import ModelSaver

LOGGER = logging.getLogger(__name__)


class ModelSaverCallback(Callback):
    """Similar to ModelCheckpoint but leaves the actual saving to the save_fn.
    """
    def __init__(
            self,
            save_fn: callable = None,
            monitor: str = 'val_loss',
            mode: str = 'auto',
            period: int = 1,
            save_best_only: bool = False,
            save_kwargs: dict = None):
        super().__init__()
        self.monitor = monitor
        self.save_fn = save_fn
        self.period = period
        self.save_best_only = save_best_only
        self.save_kwargs = save_kwargs or {}
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                'ModelCheckpoint mode %s is unknown, '
                'fallback to auto mode.' % (mode),
                RuntimeWarning
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def _save(self, epoch, logs=None):
        self.save_fn(epoch=epoch, logs=logs, **self.save_kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning
                    )
                else:
                    if self.monitor_op(current, self.best):
                        LOGGER.info(
                            'Epoch %05d: %s improved from %0.5f to %0.5f',
                            epoch + 1, self.monitor, self.best, current
                        )
                        self.best = current
                        self._save(epoch=epoch, logs=logs)
                    else:
                        LOGGER.info(
                            'Epoch %05d: %s did not improve from %0.5f',
                            epoch + 1, self.monitor, self.best
                        )
            else:
                self._save(epoch=epoch, logs=logs)


class ModelWithMetadataCheckpoint(ModelSaverCallback):
    """Similar to ModelCheckpoint but saves model metadata such as the config.
    """
    def __init__(
            self, base_path: str,
            model_saver: ModelSaver,
            add_checkpoint_meta: bool = True,
            meta: dict = None,
            **kwargs):
        self.base_path = base_path
        self.model_saver = model_saver
        self.add_checkpoint_meta = add_checkpoint_meta
        self.meta = meta
        super().__init__(save_fn=self._save_model, **kwargs)

    def _get_meta(self, epoch: int, logs: dict) -> dict:
        optimizer = self.model.optimizer
        optimizer_type = type(optimizer)
        optimizer_fullname = '%s.%s' % (optimizer_type.__module__, optimizer_type.__name__)
        return {
            **logs,
            **(self.meta or {}),
            'epoch': 1 + epoch,
            'optimizer': {
                'type': optimizer_fullname,
                'lr': float(K.get_value(optimizer.lr))
            }
        }

    def _save_model(self, epoch: int, logs: dict, **_):
        meta = self._get_meta(epoch=epoch, logs=logs)
        LOGGER.info('meta: %s', meta)
        base_path = self.base_path.format(epoch=epoch + 1, **logs)
        self.model_saver.save_to(base_path, model=self.model, meta=meta)
        if self.add_checkpoint_meta:
            self.model_saver.add_checkpoint_meta(
                base_path, epoch=epoch
            )
