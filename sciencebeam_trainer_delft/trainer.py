import logging

import numpy as np

from delft.sequenceLabelling.trainer import Trainer as _Trainer
from delft.sequenceLabelling.trainer import get_callbacks

from sciencebeam_trainer_delft.data_generator import DataGenerator


LOGGER = logging.getLogger(__name__)


class Trainer(_Trainer):
    def train(  # pylint: disable=arguments-differ
            self, x_train, y_train, x_valid, y_valid,
            features_train: np.array = None,
            features_valid: np.array = None):
        self.model.summary()

        if self.model_config.use_crf:
            self.model.compile(
                loss=self.model.crf.loss,
                optimizer='adam'
            )
        else:
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam'
            )
        self.model = self.train_model(
            self.model, x_train, y_train, x_valid, y_valid,
            self.training_config.max_epoch,
            features_train=features_train, features_valid=features_valid
        )

    def train_model(  # pylint: disable=arguments-differ
            self, local_model,
            x_train, y_train,
            x_valid=None, y_valid=None,
            max_epoch: int = 50,
            features_train: np.array = None,
            features_valid: np.array = None):
        """ parameter model local_model must be compiled before calling this method
            this model will be returned with trained weights """
        # todo: if valid set if None, create it as random segment of the shuffled train set

        if self.preprocessor.return_features and features_train is None:
            raise ValueError('features required')

        if self.training_config.early_stop:
            training_generator = DataGenerator(
                x_train, y_train,
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor,
                char_embed_size=self.model_config.char_embedding_size,
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True,
                features=features_train
            )

            validation_generator = DataGenerator(
                x_valid, y_valid,
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor,
                char_embed_size=self.model_config.char_embedding_size,
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=False,
                features=features_valid
            )

            callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=True,
                                      valid=(validation_generator, self.preprocessor))
        else:
            x_train = np.concatenate((x_train, x_valid), axis=0)
            y_train = np.concatenate((y_train, y_valid), axis=0)
            features_all = None
            if features_train is not None:
                features_all = np.concatenate((features_train, features_valid), axis=0)
            training_generator = DataGenerator(
                x_train, y_train,
                batch_size=self.training_config.batch_size, preprocessor=self.preprocessor,
                char_embed_size=self.model_config.char_embedding_size,
                max_sequence_length=self.model_config.max_sequence_length,
                embeddings=self.embeddings, shuffle=True,
                features=features_all
            )

            callbacks = get_callbacks(log_dir=self.checkpoint_path,
                                      eary_stopping=False)
        nb_workers = 6
        multiprocessing = True
        # multiple workers will not work with ELMo due to GPU memory limit (with GTX 1080Ti 11GB)
        if self.embeddings and (self.embeddings.use_ELMo or self.embeddings.use_BERT):
            # worker at 0 means the training will be executed in the main thread
            nb_workers = 0
            multiprocessing = False
            # dump token context independent data for train set, done once for the training

        local_model.fit_generator(
            generator=training_generator,
            epochs=max_epoch,
            use_multiprocessing=multiprocessing,
            workers=nb_workers,
            callbacks=callbacks
        )

        return local_model
