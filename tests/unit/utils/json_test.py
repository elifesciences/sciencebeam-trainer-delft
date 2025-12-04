from dataclasses import dataclass
from typing import Optional

import pytest
from sciencebeam_trainer_delft.utils.json import from_json, to_json


@dataclass
class DummyObjectWithoutSetState:
    str_value: Optional[str] = None
    dict_value: Optional[dict] = None


@dataclass
class DummyObjectWithSetState:
    str_value: Optional[str] = None
    dict_value: Optional[dict] = None

    def __setstate__(self, state: dict):
        self.str_value = state.get('str_value')
        self.dict_value = state.get('dict_value')


def get_full_class_name(cls) -> str:
    return cls.__module__ + '.' + cls.__name__


class TestFromJson:
    def test_should_save_and_restore_simple_object(self):
        original_object = DummyObjectWithoutSetState(
            str_value='test',
            dict_value={'a': 1}
        )
        restored_object = from_json(
            to_json(original_object)
        )
        assert restored_object == original_object

    def test_should_fail_if_class_does_not_exist(self):
        original_object = DummyObjectWithoutSetState(
            str_value='test',
            dict_value={'a': 1}
        )
        serialized_object = to_json(original_object)
        assert isinstance(serialized_object, dict)
        assert serialized_object['py/object'] == get_full_class_name(DummyObjectWithoutSetState)
        serialized_object['py/object'] = 'non.existing.Class'
        with pytest.raises(ValueError):
            from_json(serialized_object)

    def test_should_restore_simple_dict_using_provided_class(self):
        restored_object = from_json(
            {
                'str_value': 'test',
                'dict_value': {'a': 1}
            },
            DummyObjectWithSetState
        )
        assert isinstance(restored_object, DummyObjectWithSetState)
        assert restored_object.str_value == 'test'
        assert restored_object.dict_value == {'a': 1}
