import logging
from functools import wraps
from typing import Callable, TypeVar, cast


LOGGER = logging.getLogger(__name__)


T_Callable = TypeVar('T_Callable', bound=Callable)


def log_on_exception(f: T_Callable) -> T_Callable:
    """
    Wraps function to log error on exception.
    That is useful for tests that log a lot of things,
    and pytest displaying the test failure at the top of the method.
    (there doesn't seem to be an option to change that)
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            LOGGER.exception('failed due to %s', repr(e))
            raise
    return cast(T_Callable, wrapper)
