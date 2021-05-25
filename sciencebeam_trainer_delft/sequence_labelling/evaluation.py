import logging
import json
from itertools import groupby
from collections import defaultdict, OrderedDict
from typing import Iterator, List, Union, Tuple, Sequence, cast

import numpy as np

from delft.sequenceLabelling.evaluation import (
    get_entities
)

from sciencebeam_trainer_delft.utils.typing import T


LOGGER = logging.getLogger(__name__)


# mostly copied from delft/sequenceLabelling/evaluation.py
# with the following differences:
# - types are sorted
# - types are including keys from both true or prediction (not just true labels)
# - separated calculation from formatting


class EvaluationOutputFormats:
    TEXT = 'text'
    JSON = 'json'


EVALUATION_OUTPUT_FORMATS = [
    EvaluationOutputFormats.TEXT,
    EvaluationOutputFormats.JSON
]


# copied from https://stackoverflow.com/a/57915246/8676953
class NpJsonEncoder(json.JSONEncoder):
    def default(self, obj):  # pylint: disable=arguments-differ, method-hidden
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


def _get_first(some_iterator: Iterator[T]) -> T:
    return next(some_iterator)


def get_first_entities(
    y: Sequence[Union[str, Sequence[str]]],
    prefix: str = 'first_'
) -> List[Tuple[str, int, int]]:
    y_list_of_lists: Sequence[Sequence[str]] = (
        [cast(Sequence[str], y)]
        if not any(isinstance(s, list) for s in y)
        else y
    )
    offset = 0
    first_entities = []
    for seq in y_list_of_lists:
        entities = sorted(set(get_entities(seq)))
        for type_name, grouped_entities in groupby(entities, key=lambda entity: entity[0]):
            first_entity = _get_first(grouped_entities)
            first_entities.append((
                prefix + type_name,
                first_entity[1] + offset,
                first_entity[2] + offset
            ))
        offset += len(seq)
    return first_entities


class ClassificationResult:
    def __init__(
            self,
            y_true: Sequence[Union[str, Sequence[str]]],
            y_pred: Sequence[Union[str, Sequence[str]]],
            evaluate_first_entities: bool = False):
        self._y_true = y_true
        self._y_pred = y_pred
        all_true_entities = set(get_entities(y_true))
        all_pred_entities = set(get_entities(y_pred))

        if evaluate_first_entities:
            all_true_entities |= set(get_first_entities(y_true))
            all_pred_entities |= set(get_first_entities(y_pred))

        LOGGER.debug('all_true_entities: %s', all_true_entities)
        LOGGER.debug('all_pred_entities: %s', all_pred_entities)

        true_entities_by_type_name = defaultdict(set)
        pred_entities_by_type_name = defaultdict(set)
        for e in all_true_entities:
            true_entities_by_type_name[e[0]].add((e[1], e[2]))
        for e in all_pred_entities:
            pred_entities_by_type_name[e[0]].add((e[1], e[2]))

        ps, rs, f1s, s = [], [], [], []
        total_nb_correct = 0
        total_nb_pred = 0
        total_nb_true = 0
        sorted_type_names = sorted(
            set(true_entities_by_type_name.keys()) | set(pred_entities_by_type_name.keys())
        )
        # if evaluate_first_entities:
        #     for type_name in sorted_type_names.copy():
        #         first_type_name = 'first_%s' % type_name
        #         sorted_type_names.append(first_type_name)
        #         true_entities_by_type_name[first_type_name] = set(
        #             sorted(true_entities_by_type_name[type_name])[:1]
        #         )
        #         pred_entities_by_type_name[first_type_name] = set(
        #             sorted(pred_entities_by_type_name[type_name])[:1]
        #         )
        self.scores = OrderedDict()
        for type_name in sorted_type_names:
            true_entities = true_entities_by_type_name[type_name]
            pred_entities = pred_entities_by_type_name[type_name]
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

    def with_first_entities(self):
        return ClassificationResult(
            y_true=self._y_true,
            y_pred=self._y_pred,
            evaluate_first_entities=True
        )

    @property
    def f1(self):
        return self.micro_averages['f1']

    @property
    def text_formatted_report(self):
        return self.get_text_formatted_report().rstrip()

    def get_text_formatted_report(
            self,
            digits: int = 4,
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

    def get_dict_formatted_report(self):
        return {
            'scores': self.scores,
            'micro_averages': self.micro_averages
        }

    def get_json_formatted_report(self, meta: dict = None):
        dict_report = self.get_dict_formatted_report()
        if meta:
            dict_report['meta'] = meta
        return json.dumps(
            dict_report,
            indent=2,
            cls=NpJsonEncoder
        )

    def get_formatted_report(
            self,
            output_format: str = EvaluationOutputFormats.TEXT,
            **kwargs):
        if output_format == EvaluationOutputFormats.TEXT:
            return self.get_text_formatted_report(**kwargs)
        if output_format == EvaluationOutputFormats.JSON:
            return self.get_json_formatted_report()
        raise ValueError('unsupported output format: %s' % output_format)


def classification_report(
        y_true: Sequence[Union[str, List[str]]],
        y_pred: Sequence[Union[str, List[str]]],
        digits: int = 2,
        exclude_no_support: bool = False):
    return ClassificationResult(
        y_true=y_true,
        y_pred=y_pred
    ).get_formatted_report(
        digits=digits,
        exclude_no_support=exclude_no_support
    )
