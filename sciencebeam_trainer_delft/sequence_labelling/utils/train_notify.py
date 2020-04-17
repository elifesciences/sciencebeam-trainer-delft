import argparse
import logging

import requests

from sciencebeam_trainer_delft.sequence_labelling.evaluation import (
    ClassificationResult
)


LOGGER = logging.getLogger(__name__)


DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT = 'Model training complete'

DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT = (
    'Model training complete, f1: `{classification_result.f1:.4f}`\nmodel_path: `{model_path}`'
)

DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT = (
    'Model training failed due to: `{error}`\nmodel_path: `{model_path}`'
)


def get_rendered_notification_message(message_format: str, **kwargs):
    return message_format.format(**kwargs)


def get_fallback_notification_message(message_format: str, conversion_error: str, args: dict):
    return 'failed to format %r due to %s (args: %s)' % (message_format, conversion_error, args)


def safe_rendered_notification_message(message_format: str, **kwargs):
    try:
        return get_rendered_notification_message(message_format, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.warning('failed to convert message due to: %s', e, exc_info=1)
        return get_fallback_notification_message(message_format, str(e), kwargs)


class TrainNotificationManager:
    def __init__(
            self,
            notification_url: str,
            notification_train_success_message: str,
            notification_train_eval_success_message: str,
            notification_error_message: str):
        self.notification_url = notification_url
        self.notification_train_success_message = notification_train_success_message
        self.notification_train_eval_success_message = notification_train_eval_success_message
        self.notification_error_message = notification_error_message

    def send_notification(self, message_format: str, **kwargs):
        message = safe_rendered_notification_message(message_format, **kwargs)
        LOGGER.info('send_notification: %s (url: %s)', message, self.notification_url)
        if not message or not self.notification_url:
            return
        data = {
            'text': message
        }
        requests.post(self.notification_url, json=data)

    def notify_error(self, model_path: str, error: str):
        self.send_notification(
            self.notification_error_message,
            model_path=model_path,
            error=error
        )

    def notify_success(self, model_path: str, classification_result: ClassificationResult = None):
        if classification_result is None:
            self.send_notification(
                self.notification_train_success_message,
                model_path=model_path
            )
        else:
            self.send_notification(
                self.notification_train_eval_success_message,
                model_path=model_path,
                classification_result=classification_result
            )


def add_train_notification_arguments(parser: argparse.ArgumentParser):
    notification_group = parser.add_argument_group('notification')
    notification_group.add_argument(
        "--notification-url",
        help="A URL to post to on success error (e.g. a Slack Webhook URL)"
    )
    notification_group.add_argument(
        "--notification-train-success-message",
        default=DEFAULT_TRAIN_SUCCESS_MESSAGE_FORMAT,
        help="Model training success"
    )
    notification_group.add_argument(
        "--notification-train-eval-success-message",
        default=DEFAULT_TRAIN_EVAL_SUCCESS_MESSAGE_FORMAT,
        help="Model training and evaluation success"
    )
    notification_group.add_argument(
        "--notification-error-message",
        default=DEFAULT_TRAIN_ERROR_MESSAGE_FORMAT,
        help="Model training failed: {error}"
    )


def get_train_notification_manager(args: argparse.Namespace) -> TrainNotificationManager:
    return TrainNotificationManager(
        notification_url=args.notification_url,
        notification_train_success_message=args.notification_train_success_message,
        notification_train_eval_success_message=args.notification_train_eval_success_message,
        notification_error_message=args.notification_error_message
    )


def notify_train_success(
        train_notification_manager: TrainNotificationManager = None,
        **kwargs):
    if train_notification_manager is not None:
        train_notification_manager.notify_success(**kwargs)


def notify_train_error(
        train_notification_manager: TrainNotificationManager = None,
        **kwargs):
    if train_notification_manager is not None:
        train_notification_manager.notify_error(**kwargs)
