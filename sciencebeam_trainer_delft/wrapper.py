import numpy as np

from delft.sequenceLabelling.wrapper import Sequence as _Sequence

from sciencebeam_trainer_delft.trainer import Trainer
from sciencebeam_trainer_delft.models import get_model
from sciencebeam_trainer_delft.preprocess import WordPreprocessor


def prepare_preprocessor(X, y, model_config):
    p = WordPreprocessor(max_char_length=model_config.max_char_length)
    p.fit(X, y)
    return p


class Sequence(_Sequence):
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        # TBD if valid is None, segment train to get one
        x_all = np.concatenate((x_train, x_valid), axis=0)
        y_all = np.concatenate((y_train, y_valid), axis=0)
        self.p = prepare_preprocessor(x_all, y_all, self.model_config)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        self.model_config.case_vocab_size = len(self.p.vocab_case)

        self.model = get_model(self.model_config, self.p, len(self.p.vocab_tag))
        trainer = Trainer(
            self.model,
            self.models,
            self.embeddings,
            self.model_config,
            self.training_config,
            checkpoint_path=self.log_dir,
            preprocessor=self.p
        )
        trainer.train(x_train, y_train, x_valid, y_valid)
        if self.embeddings.use_ELMo:
            self.embeddings.clean_ELMo_cache()
        if self.embeddings.use_BERT:
            self.embeddings.clean_BERT_cache()
