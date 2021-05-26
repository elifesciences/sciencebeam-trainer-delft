from typing import List, Dict, Optional, Tuple


def parse_comma_separated_str(s: str) -> List[str]:
    if not s:
        return []
    return [item.strip() for item in s.split(',')]


def parse_number_range(expr: str) -> List[int]:
    fragments = expr.split('-')
    if len(fragments) == 1:
        return [int(expr)]
    if len(fragments) == 2:
        return list(range(int(fragments[0]), int(fragments[1]) + 1))
    raise ValueError('invalid number range: %s' % fragments)


def parse_number_ranges(expr: str) -> List[int]:
    if not expr:
        return []
    numbers = []
    for fragment in expr.split(','):
        numbers.extend(parse_number_range(fragment))
    return numbers


def parse_key_value(expr: str) -> Tuple[str, str]:
    key, value = expr.split('=', maxsplit=1)
    return key.strip(), value.strip()


def parse_dict(expr: str, delimiter: str = '|') -> Dict[str, str]:
    if not expr:
        return {}
    d = {}
    for fragment in expr.split(delimiter):
        key, value = parse_key_value(fragment)
        d[key] = value
    return d


def merge_dicts(dict_list: List[dict]) -> dict:
    result = {}
    for d in dict_list:
        result.update(d)
    return result


def str_to_bool(value: str, default_value: Optional[bool] = None) -> Optional[bool]:
    if not value:
        return default_value
    if value.lower() in {'true', 't', 'yes', '1'}:
        return True
    if value.lower() in {'false', 'f', 'no', '0'}:
        return False
    raise ValueError('invalid boolean value: %r' % value)
