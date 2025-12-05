import logging
from collections import OrderedDict
from typing import List

import numpy as np
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)

from sciencebeam_trainer_delft.text_classification.typing import (
    T_Batch_Text_Classes_Array
)


LOGGER = logging.getLogger(__name__)


class ClassificationResult:
    def __init__(
        self,
        y_true: T_Batch_Text_Classes_Array,
        y_pred: T_Batch_Text_Classes_Array,
        label_names: List[str]
    ):
        y_true_array: np.ndarray = np.asarray(y_true)
        y_pred_array: np.ndarray = np.asarray(y_pred)
        LOGGER.info('y_true: %s', y_true)
        LOGGER.info('y_pred: %s', y_pred)
        self.scores = OrderedDict()
        for j, label_name in enumerate(label_names):
            labels = [0, 1]
            y_true_class = y_true_array[:, j]
            y_pred_class = y_pred_array[:, j]
            y_true_binary_class = y_true_array[:, j] >= 0.5
            y_pred_binary_class = y_pred_array[:, j] >= 0.5
            loss = log_loss(y_true_class, y_pred_class, labels=labels)
            precision = precision_score(y_true_binary_class, y_pred_binary_class, zero_division=0)
            recall = recall_score(y_true_binary_class, y_pred_binary_class, zero_division=0)
            f1 = f1_score(y_true_binary_class, y_pred_binary_class, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_true_class, y_pred_class)
            except ValueError as e:
                LOGGER.warning('could not calculate roc (index=%d): %s', j, e)
                roc_auc = np.nan
            self.scores[label_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'loss': loss,
                'roc_auc': roc_auc,
                'support': np.sum(y_true_binary_class)
            }
        self.macro_averages = {
            'precision': np.mean([score['precision'] for score in self.scores.values()]),
            'recall': np.mean([score['recall'] for score in self.scores.values()]),
            'f1': np.mean([score['f1'] for score in self.scores.values()]),
            'loss': np.mean([score['loss'] for score in self.scores.values()]),
            'roc_auc': np.mean([score['roc_auc'] for score in self.scores.values()]),
            'support': np.sum([score['support'] for score in self.scores.values()]),
        }

    @property
    def text_formatted_report(self):
        return self.get_text_formatted_report().rstrip()

    def get_text_formatted_report(
            self,
            digits: int = 4,
            exclude_no_support: bool = False):
        name_width = max(map(len, self.scores.keys()))

        last_line_heading = 'all (macro avg. / mean)'
        width = max(name_width, len(last_line_heading), digits)

        headers = ["precision", "recall", "f1-score", "support", "roc_auc"]
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'

        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}' + u' {:>9.{digits}f}\n'

        for type_name in sorted(self.scores.keys()):
            item_scores = self.scores[type_name]
            if exclude_no_support and not item_scores['support']:
                continue

            report += row_fmt.format(
                *[
                    type_name,
                    item_scores['precision'],
                    item_scores['recall'],
                    item_scores['f1'],
                    item_scores['support'],
                    item_scores['roc_auc']
                ],
                width=width,
                digits=digits
            )

        report += u'\n'

        report += row_fmt.format(
            *[
                last_line_heading,
                self.macro_averages['precision'],
                self.macro_averages['recall'],
                self.macro_averages['f1'],
                self.macro_averages['support'],
                self.macro_averages['roc_auc']
            ],
            width=width,
            digits=digits
        )

        return report
