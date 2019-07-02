from typing import Dict, List, Type

from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import (
    Dense, LSTM, Bidirectional, Embedding, Input, Dropout,
    TimeDistributed
)

import delft.sequenceLabelling.wrapper
from delft.utilities.layers import ChainCRF
from delft.sequenceLabelling.models import BaseModel
from delft.sequenceLabelling.models import get_model as _get_model


class CustomModel(BaseModel):
    def __init__(self, config, ntags, require_casing: bool = False, use_crf: bool = False):
        super().__init__(config, ntags)
        self.require_casing = require_casing
        self.use_crf = use_crf


# renamed copy of BidLSTM_CRF to demonstrate a custom model
class CustomBidLSTM_CRF(CustomModel):
    """
    A Keras implementation of BidLSTM-CRF for sequence labelling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config, ntags=None):
        super().__init__(config, ntags, require_casing=False, use_crf=True)

        # build input, directly feed with word embedding by the data generator
        word_input = Input(shape=(None, config.word_embedding_size), name='word_input')

        # build character based embedding
        char_input = Input(shape=(None, config.max_char_length), dtype='int32', name='char_input')
        char_embeddings = TimeDistributed(Embedding(
            input_dim=config.char_vocab_size,
            output_dim=config.char_embedding_size,
            # mask_zero=True,
            # embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
            name='char_embeddings'
        ))(char_input)

        chars = TimeDistributed(
            Bidirectional(LSTM(config.num_char_lstm_units, return_sequences=False))
        )(char_embeddings)

        # length of sequence not used for the moment (but used for f1 communication)
        length_input = Input(batch_shape=(None, 1), dtype='int32', name='length_input')

        # combine characters and word embeddings
        x = Concatenate()([word_input, chars])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(
            units=config.num_word_lstm_units,
            return_sequences=True,
            recurrent_dropout=config.recurrent_dropout
        ))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        self.model = Model(inputs=[word_input, char_input, length_input], outputs=[pred])
        self.config = config


DEFAULT_MODEL_NAMES = [
    'BidLSTM_CRF', 'BidLSTM_CNN', 'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CRF_CASING'
]

MODEL_MAP: Dict[str, Type[CustomModel]] = {
    'CustomBidLSTM_CRF': CustomBidLSTM_CRF
}


def register_model(name: str, model_class: Type[CustomModel]):
    MODEL_MAP[name] = model_class


def get_model(config, preprocessor, ntags=None):
    model_class = MODEL_MAP.get(config.model_type)
    if not model_class:
        return _get_model(config, preprocessor, ntags=ntags)
    model: CustomModel = model_class(config, ntags=ntags)
    config.use_crf = model.use_crf
    preprocessor.require_casing = model.require_casing
    return model


def get_model_names() -> List[str]:
    return sorted(set(DEFAULT_MODEL_NAMES) | set(MODEL_MAP.keys()))


def patch_get_model():
    delft.sequenceLabelling.wrapper.get_model = get_model