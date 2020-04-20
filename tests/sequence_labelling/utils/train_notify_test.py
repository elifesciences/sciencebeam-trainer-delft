import argparse

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    ClassificationResult
)

from sciencebeam_trainer_delft.sequence_labelling.utils.train_notify import (
    get_rendered_notification_message,
    add_train_notification_arguments,
    get_train_notification_manager,
    DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT,
    DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT,
    DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT
)


class TestGetRenderedNotificationMessage:
    def test_should_return_static_message(self):
        assert get_rendered_notification_message('test') == 'test'

    def test_should_return_replace_placeholder(self):
        classification_result = ClassificationResult(['B-DUMMY'], ['B-DUMMY'])
        assert get_rendered_notification_message(
            'f1: {classification_result.f1}',
            classification_result=classification_result
        ) == 'f1: 1.0'

    def test_should_not_fail_using_default_train_message(self):
        get_rendered_notification_message(
            DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT,
            last_checkpoint_path=None,
            model_path='model_path'
        )

    def test_should_not_fail_using_default_train_eval_message(self):
        classification_result = ClassificationResult(['B-DUMMY'], ['B-DUMMY'])
        get_rendered_notification_message(
            DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT,
            model_path='model_path',
            last_checkpoint_path=None,
            classification_result=classification_result
        )

    def test_should_not_fail_using_default_train_error_message(self):
        get_rendered_notification_message(
            DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT,
            model_path='model_path',
            error='error'
        )


class TestGetTrainNotificationManager:
    def test_should_be_able_to_get_train_notification_manager_with_defaults(self):
        parser = argparse.ArgumentParser()
        add_train_notification_arguments(parser)
        args = parser.parse_args([])
        train_notification_manager = get_train_notification_manager(args)
        assert train_notification_manager is not None
        train_notification_manager.notify_success(model_path='model_path')
        train_notification_manager.notify_error(model_path='model_path', error='error')
