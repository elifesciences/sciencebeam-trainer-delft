import logging

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    get_first_entities,
    ClassificationResult
)


LOGGER = logging.getLogger(__name__)


class TestGetFirstEntities:
    def test_should_extract_first_entities_from_multiple_sequences(self):
        y = [
            ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O'],
            ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC']
        ]
        seq2_offset = len(y[0])
        assert set(get_first_entities(y)) == {
            ('first_MISC', 3, 4),
            ('first_PER', 0 + seq2_offset, 1 + seq2_offset),
            ('first_MISC', 5 + seq2_offset, 6 + seq2_offset)
        }


class TestClassificationResult:
    def test_should_generate_evaluation_report(self):
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [
            ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'],
            ['B-PER', 'I-PER', 'O']
        ]
        result = ClassificationResult(y_true=y_true, y_pred=y_pred)
        scores = result.scores
        assert scores.keys() == {'MISC', 'PER'}
        assert scores['MISC'] == {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'support': 1
        }
        assert scores['PER'] == {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'support': 1
        }
        assert result.micro_averages == {
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'support': 2
        }
        formatted_report = result.get_formatted_report(digits=2)
        LOGGER.debug('formatted_report:\n%s', formatted_report)
        LOGGER.debug('formatted_report lines: %s', formatted_report.splitlines())
        assert formatted_report.splitlines() == [
            '                  precision    recall  f1-score   support',
            '',
            '            MISC       0.00      0.00      0.00         1',
            '             PER       1.00      1.00      1.00         1',
            '',
            'all (micro avg.)       0.50      0.50      0.50         2'
        ]

    def test_should_add_first_entities_if_enabled(self):
        y_true = [
            ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O'],
            ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER']
        ]
        y_pred = [
            ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'B-MISC', 'O'],
            ['B-PER', 'I-PER', 'B-PER', 'I-PER', 'O']
        ]
        result = ClassificationResult(
            y_true=y_true, y_pred=y_pred
        ).with_first_entities()
        scores = result.scores
        assert scores.keys() == {'MISC', 'PER', 'first_MISC', 'first_PER'}
        assert scores['first_MISC'] == {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'support': 1
        }
        assert scores['first_PER'] == {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'support': 1
        }
        assert scores['MISC'] == {
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'support': 2
        }
        assert scores['PER'] == {
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'support': 2
        }
        assert result.micro_averages == {
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'support': 6
        }
        formatted_report = result.get_formatted_report(digits=2)
        LOGGER.debug('formatted_report:\n%s', formatted_report)
        LOGGER.debug('formatted_report lines: %s', formatted_report.splitlines())
        assert formatted_report.splitlines() == [
            '                  precision    recall  f1-score   support',
            '',
            '            MISC       0.50      0.50      0.50         2',
            '             PER       0.50      0.50      0.50         2',
            '      first_MISC       0.00      0.00      0.00         1',
            '       first_PER       1.00      1.00      1.00         1',
            '',
            'all (micro avg.)       0.50      0.50      0.50         6'
        ]

    def test_should_first_entities_entities_from_multiple_sequences(self):
        y_true = [
            ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'O'],
            ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC']
        ]
        y_pred = [
            ['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'B-MISC', 'O'],
            ['B-PER', 'I-PER', 'B-PER', 'I-PER', 'O', 'B-MISC', 'I-MISC']
        ]
        result = ClassificationResult(
            y_true=y_true, y_pred=y_pred
        ).with_first_entities()
        scores = result.scores
        assert scores.keys() == {'MISC', 'PER', 'first_MISC', 'first_PER'}
        assert scores['first_MISC'] == {
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'support': 2
        }
        assert scores['first_PER'] == {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'support': 1
        }
