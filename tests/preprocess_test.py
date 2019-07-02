import logging

from sciencebeam_trainer_delft.preprocess import WordPreprocessor


LOGGER = logging.getLogger(__name__)


class TestWordPreprocessor:
    def test_should_be_able_to_instantiate_with_default_values(self):
        WordPreprocessor()

    def test_should_fit_empty_dataset(self):
        preprocessor = WordPreprocessor()
        preprocessor.fit([], [])

    def test_should_fit_single_word_dataset(self):
        preprocessor = WordPreprocessor()
        X = [['Word1']]
        y = [['label1']]
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)
        LOGGER.debug('vocab_char: %s', preprocessor.vocab_char)
        LOGGER.debug('vocab_case: %s', preprocessor.vocab_case)
        LOGGER.debug('vocab_tag: %s', preprocessor.vocab_tag)
        LOGGER.debug('X_transformed: %s', X_transformed)
        LOGGER.debug('y_transformed: %s', y_transformed)
        for c in 'Word1':
            assert c in preprocessor.vocab_char
        for case in {'numeric', 'allLower', 'allUpper', 'initialUpper'}:
            assert case in preprocessor.vocab_case
        assert 'label1' in preprocessor.vocab_tag
        assert len(X_transformed) == 1
        assert len(y_transformed) == 1

    def test_should_be_able_to_inverse_transform_label(self):
        preprocessor = WordPreprocessor()
        X = [['Word1']]
        y = [['label1']]
        _, y_transformed = preprocessor.fit_transform(X, y)
        y_inverse = preprocessor.inverse_transform(y_transformed[0])
        assert y_inverse == y[0]
