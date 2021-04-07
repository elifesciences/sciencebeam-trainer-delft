import logging
from pathlib import Path
from typing import List, Optional, NamedTuple, Union, T

from sciencebeam_trainer_delft.sequence_labelling.tools.checkpoints import (
    get_checkpoints_json
)


LOGGER = logging.getLogger(__name__)


class CheckPoint(NamedTuple):
    path: str
    epoch: int


def get_sorted_checkpoint_json_list(checkpoints_json: dict) -> List[dict]:
    return sorted(
        checkpoints_json.get('checkpoints', []),
        key=lambda checkpoint: checkpoint['path']
    )


def get_checkpoint_for_json(checkpoint_json: Optional[dict]) -> Optional[CheckPoint]:
    if not checkpoint_json:
        return None
    epoch = checkpoint_json.get('epoch')
    assert epoch
    return CheckPoint(
        path=checkpoint_json.get('path'),
        epoch=epoch
    )


def get_last_or_none(a_list: List[T]) -> Optional[T]:
    try:
        return a_list[-1]
    except IndexError:
        return None


class CheckPoints:
    def __init__(self, log_dir: Union[str, Path]):
        assert log_dir
        self.log_dir = str(log_dir)
        self._checkpoints_json: dict = None

    def _load_checkpoints_json(self) -> dict:
        try:
            return get_checkpoints_json(
                self.log_dir
            )
        except FileNotFoundError:
            return {}

    @property
    def checkpoints_json(self) -> dict:
        if self._checkpoints_json is None:
            self._checkpoints_json = self._load_checkpoints_json()
        return self._checkpoints_json

    @property
    def latest_checkpoint(self) -> Optional[CheckPoint]:
        return get_checkpoint_for_json(
            get_last_or_none(
                get_sorted_checkpoint_json_list(self.checkpoints_json)
            )
        )

    @property
    def latest_checkpoint_url(self) -> Optional[str]:
        latest_checkpoint = self.latest_checkpoint
        return latest_checkpoint.path if latest_checkpoint else None


class ResumeTrainModelParams(NamedTuple):
    model_path: str
    initial_epoch: int


def get_resume_train_model_params(
    log_dir: Optional[str],
    auto_resume: bool = True,
    resume_train_model_path: Optional[str] = None,
    initial_epoch: Optional[int] = None
) -> Optional[ResumeTrainModelParams]:
    if auto_resume and log_dir:
        latest_checkpoint = CheckPoints(log_dir=log_dir).latest_checkpoint
        if latest_checkpoint:
            LOGGER.info('auto resuming using latest checkpoint: %r', latest_checkpoint)
            return ResumeTrainModelParams(
                model_path=latest_checkpoint.path,
                initial_epoch=latest_checkpoint.epoch
            )
    if resume_train_model_path:
        LOGGER.info('using passed in resume train model path: %r', resume_train_model_path)
        return ResumeTrainModelParams(
            model_path=resume_train_model_path,
            initial_epoch=initial_epoch or 0
        )
    return None
