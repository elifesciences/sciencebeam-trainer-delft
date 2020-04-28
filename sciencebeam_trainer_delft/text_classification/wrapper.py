import os

from delft.textClassification.models import getModel

from delft.textClassification.wrapper import (
    Classifier as _Classifier
)

from sciencebeam_trainer_delft.embedding.embedding import Embeddings
from sciencebeam_trainer_delft.text_classification.config import ModelConfig


class Classifier(_Classifier):
    def save_to(self, model_path: str):
        # create subfolder for the model if not already exists
        directory = model_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.model_config.save(os.path.join(directory, self.config_file))
        print('model config file saved')

        if self.model_config.fold_number is 1:
            if self.model is not None:
                self.model.save(os.path.join(
                    directory,
                    self.model_config.model_type + "." + self.weight_file
                ))
                print('model saved')
            else:
                print('Error: model has not been built')
        else:
            if self.models is None:
                print('Error: nfolds models have not been built')
            else:
                for i in range(0, self.model_config.fold_number):
                    self.models[i].save(os.path.join(
                        directory,
                        self.model_config.model_type + ".model{0}_weights.hdf5".format(i)
                    ))
                print('nfolds model saved')

    def load_from(self, model_path: str):
        self.model_config = ModelConfig.load(
            os.path.join(model_path, self.config_file)
        )

        # load embeddings
        self.embeddings = Embeddings(
            self.model_config.embeddings_name,
            use_ELMo=self.model_config.use_ELMo,
            use_BERT=self.model_config.use_BERT
        )
        self.model_config.word_embedding_size = self.embeddings.embed_size

        self.model = getModel(self.model_config, self.training_config)
        if self.model_config.fold_number is 1:
            self.model.load_weights(os.path.join(
                model_path,
                self.model_config.model_type + "." + self.weight_file
            ))
        else:
            self.models = []
            for i in range(0, self.model_config.fold_number):
                local_model = getModel(self.model_config, self.training_config)
                local_model.load_weights(os.path.join(
                    model_path,
                    self.model_config.model_type + ".model{0}_weights.hdf5".format(i)
                ))
                self.models.append(local_model)
