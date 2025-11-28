import jsonpickle


def to_json(obj, plain_json: bool = False):
    return jsonpickle.Pickler(
        unpicklable=not plain_json,
        keys=True
    ).flatten(obj)


def from_json(json, default_class=None):
    result = jsonpickle.Unpickler(keys=True).restore(json)
    if not isinstance(result, dict) or default_class is None:
        return result
    obj = default_class()
    obj.__setstate__(result)
    return obj
