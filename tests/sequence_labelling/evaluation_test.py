import logging

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    ClassificationResult
)


LOGGER = logging.getLogger(__name__)


class TestClassificationResult:
    def test_(self):
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
