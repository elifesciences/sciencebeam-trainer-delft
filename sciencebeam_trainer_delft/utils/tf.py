try:
    from tensorflow import __version__ as tf_version
    from tensorflow.python.client import device_lib as tf_device_lib
except ImportError:
    tf_version = None
    tf_device_lib = None


def get_tf_info():
    return {
        'tf_version': tf_version,
        'tf_device_lib': tf_device_lib.list_local_devices() if tf_device_lib else None
    }
