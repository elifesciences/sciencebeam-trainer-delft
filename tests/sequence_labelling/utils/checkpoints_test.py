import json
from pathlib import Path

import pytest

from sciencebeam_trainer_delft.sequence_labelling.utils.checkpoints import (
    CheckPoints,
    get_resume_train_model_params
)


class TestCheckPoints:
    def test_should_raise_exception_without_log_dir(self):
        with pytest.raises(AssertionError):
            CheckPoints(None)

    def test_should_accept_log_dir_str(self, tmp_path: Path):
        CheckPoints(str(tmp_path))

    def test_should_accept_log_dir_path(self, tmp_path: Path):
        CheckPoints(tmp_path)

    def test_should_return_none_latest_checkpoint_without_checkpoints_json(self, tmp_path: Path):
        assert CheckPoints(tmp_path).latest_checkpoint is None

    def test_should_return_get_latest_checkpoint(self, tmp_path: Path):
        latest_checkpoint_path = str(tmp_path / 'epoch-00001')
        (tmp_path / 'checkpoints.json').write_text(json.dumps({
            'checkpoints': [{
                'path': latest_checkpoint_path,
                'epoch': 1
            }]
        }))
        latest_checkpoint = CheckPoints(tmp_path).latest_checkpoint
        assert latest_checkpoint
        assert latest_checkpoint.path == latest_checkpoint_path

    def test_should_return_get_latest_checkpoint_with_highest_epoch(self, tmp_path: Path):
        checkpoint_path_with_highest_epoch = str(tmp_path / 'epoch-00002')
        latest_checkpoint_path = str(tmp_path / 'epoch-00001')
        (tmp_path / 'checkpoints.json').write_text(json.dumps({
            'checkpoints': [{
                'path': checkpoint_path_with_highest_epoch,
                'epoch': 2
            }, {
                'path': latest_checkpoint_path,
                'epoch': 1
            }]
        }))
        latest_checkpoint = CheckPoints(tmp_path).latest_checkpoint
        assert latest_checkpoint
        assert latest_checkpoint.path == checkpoint_path_with_highest_epoch


class TestGetResumeTrainModelParams:
    def test_should_return_none_without_log_dir_and_resume_train_model_path(self):
        assert get_resume_train_model_params(
            log_dir=None,
            auto_resume=True,
            resume_train_model_path=None
        ) is None

    def test_should_return_passed_in_resume_train_model_path(
        self,
        tmp_path: Path
    ):
        resume_train_model_path = str(tmp_path / 'epoch-00001')
        result = get_resume_train_model_params(
            log_dir=None,
            auto_resume=True,
            resume_train_model_path=resume_train_model_path
        )
        assert result
        assert result.model_path == resume_train_model_path

    def test_should_return_latest_checkpoint_if_auto_resume_is_enabled(
        self,
        tmp_path: Path
    ):
        resume_train_model_path = str(tmp_path / 'epoch-00001')
        latest_checkpoint_path = str(tmp_path / 'epoch-00002')
        (tmp_path / 'checkpoints.json').write_text(json.dumps({
            'checkpoints': [{
                'path': resume_train_model_path,
                'epoch': 1
            }, {
                'path': latest_checkpoint_path,
                'epoch': 2
            }]
        }))
        result = get_resume_train_model_params(
            log_dir=str(tmp_path),
            auto_resume=True,
            initial_epoch=123,
            resume_train_model_path=resume_train_model_path
        )
        assert result
        assert result.model_path == latest_checkpoint_path
        assert result.initial_epoch == 2

    def test_should_ignore_latest_checkpoint_if_auto_resume_is_disabled(
        self,
        tmp_path: Path
    ):
        resume_train_model_path = str(tmp_path / 'epoch-00001')
        latest_checkpoint_path = str(tmp_path / 'epoch-00002')
        (tmp_path / 'checkpoints.json').write_text(json.dumps({
            'checkpoints': [{
                'path': resume_train_model_path,
                'epoch': 1
            }, {
                'path': latest_checkpoint_path,
                'epoch': 2
            }]
        }))
        result = get_resume_train_model_params(
            log_dir=str(tmp_path),
            auto_resume=False,
            initial_epoch=123,
            resume_train_model_path=resume_train_model_path
        )
        assert result
        assert result.model_path == resume_train_model_path
        assert result.initial_epoch == 123

    def test_should_load_meta(
        self,
        tmp_path: Path
    ):
        latest_checkpoint_path = tmp_path / 'epoch-00001'
        latest_checkpoint_path.mkdir()
        meta = {
            'prop1': 'value1'
        }
        (latest_checkpoint_path / 'meta.json').write_text(json.dumps(meta))
        (tmp_path / 'checkpoints.json').write_text(json.dumps({
            'checkpoints': [{
                'path': str(latest_checkpoint_path),
                'epoch': 1
            }]
        }))
        result = get_resume_train_model_params(
            log_dir=str(tmp_path),
            auto_resume=True,
            resume_train_model_path=None
        )
        assert result
        assert result.model_path == str(latest_checkpoint_path)
        assert result.initial_epoch == 1
        assert result.initial_meta == meta
