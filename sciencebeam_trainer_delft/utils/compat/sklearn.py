# type: ignore
# pylint: disable=unused-import

try:
    import sklearn.feature_extraction.dict_vectorizer  # noqa
except ImportError:
    import sys
    from sklearn.feature_extraction import _dict_vectorizer
    sys.modules['sklearn.feature_extraction.dict_vectorizer'] = (
        _dict_vectorizer
    )
