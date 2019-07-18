import os

from delft.utilities.Embeddings import Embeddings as _Embeddings


class Embeddings(_Embeddings):
    def make_embeddings_lmdb(self, *args, **kwargs):  # pylint: disable=arguments-differ
        try:
            super().make_embeddings_lmdb(*args, **kwargs)
        except FileNotFoundError as e:
            abs_path = os.path.abspath(e.filename)
            raise FileNotFoundError('file not found: %s (%s)' % (e.filename, abs_path)) from e
