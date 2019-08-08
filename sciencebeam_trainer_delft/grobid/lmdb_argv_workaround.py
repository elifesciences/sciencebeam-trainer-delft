import sys


# see https://github.com/jnwatson/py-lmdb/pull/207
try:
    if not sys.argv:
        sys.argv = ['']
except AttributeError:
    pass
