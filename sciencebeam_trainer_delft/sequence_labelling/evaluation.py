from collections import defaultdict, OrderedDict
from typing import List, Union

import numpy as np

from delft.sequenceLabelling.evaluation import (
    get_entities
)


# mostly copied from delft/sequenceLabelling/evaluation.py
# with the following differences:
# - types are sorted
# - types are including keys from both true or prediction (not just true labels)
# - separated calculation from formatting

class ClassificationResult:
    def __init__(
            self,
            y_true: List[Union[str, List[str]]],
            y_pred: List[Union[str, List[str]]]):
        true_entities = set(get_entities(y_true))
        pred_entities = set(get_entities(y_pred))

        d1 = defaultdict(set)
        d2 = defaultdict(set)
        for e in true_entities:
            d1[e[0]].add((e[1], e[2]))
        for e in pred_entities:
            d2[e[0]].add((e[1], e[2]))

        ps, rs, f1s, s = [], [], [], []
        total_nb_correct = 0
        total_nb_pred = 0
        total_nb_true = 0
        sorted_type_names = sorted(set(d1.keys()) | set(d2.keys()))
        self.scores = OrderedDict()
        for type_name in sorted_type_names:
            true_entities = d1[type_name]
            pred_entities = d2[type_name]
            nb_correct = len(true_entities & pred_entities)
            nb_pred = len(pred_entities)
            nb_true = len(true_entities)

            precision = nb_correct / nb_pred if nb_pred > 0 else 0
            recall = nb_correct / nb_true if nb_true > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

            self.scores[type_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': nb_true
            }

            ps.append(precision)
            rs.append(recall)
            f1s.append(f1)
            s.append(nb_true)

            total_nb_correct += nb_correct
            total_nb_true += nb_true
            total_nb_pred += nb_pred

        # micro average
        micro_precision = total_nb_correct / total_nb_pred if total_nb_pred > 0 else 0
        micro_recall = total_nb_correct / total_nb_true if total_nb_true > 0 else 0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall > 0
            else 0
        )
        self.micro_averages = {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'support': np.sum(s)
        }

    def get_formatted_report(
            self,
            digits: int = 2,
            exclude_no_support: bool = False):
        name_width = max(map(len, self.scores.keys()))

        last_line_heading = 'all (micro avg.)'
        width = max(name_width, len(last_line_heading), digits)

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'

        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

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
                    item_scores['support']
                ],
                width=width,
                digits=digits
            )

        report += u'\n'

        report += row_fmt.format(
            *[
                last_line_heading,
                self.micro_averages['precision'],
                self.micro_averages['recall'],
                self.micro_averages['f1'],
                self.micro_averages['support']
            ],
            width=width,
            digits=digits
        )

        return report


def classification_report(
        y_true: List[Union[str, List[str]]],
        y_pred: List[Union[str, List[str]]],
        digits: int = 2,
        exclude_no_support: bool = False):
    return ClassificationResult(
        y_true=y_true,
        y_pred=y_pred
    ).get_formatted_report(
        digits=digits,
        exclude_no_support=exclude_no_support
    )
