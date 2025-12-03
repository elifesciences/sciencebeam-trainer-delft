from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Label_List,
    T_Batch_Token_Array,
    T_Batch_Features_Array,
    T_Batch_Label_Array
)


class DatasetTransformer(ABC):
    @abstractmethod
    def fit_transform(
        self,
        x: T_Batch_Token_Array,
        y: Optional[T_Batch_Label_Array],
        features: Optional[T_Batch_Features_Array]
    ) -> Tuple[
        T_Batch_Token_Array,
        Optional[T_Batch_Label_Array],
        Optional[T_Batch_Features_Array]
    ]:
        pass

    @abstractmethod
    def inverse_transform(
        self,
        x: Optional[T_Batch_Token_Array],
        y: Optional[T_Batch_Label_Array],
        features: Optional[T_Batch_Features_Array]
    ) -> Tuple[
        Optional[T_Batch_Token_Array],
        Optional[Union[T_Batch_Label_Array, T_Batch_Label_List]],
        Optional[T_Batch_Features_Array]
    ]:
        pass

    def fit_transform_x_and_features(
        self,
        x: T_Batch_Token_Array,
        features: Optional[T_Batch_Features_Array]
    ):
        x, _, features = self.fit_transform(x, None, features)
        return x, features

    def inverse_transform_y(
        self,
        y: T_Batch_Label_Array,
    ) -> T_Batch_Label_Array:
        _, inverse_transformed_y, _ = self.inverse_transform(None, y, None)
        assert inverse_transformed_y is not None
        return inverse_transformed_y


class DummyDatasetTransformer(DatasetTransformer):
    def fit_transform(
        self,
        x: T_Batch_Token_Array,
        y: Optional[T_Batch_Label_Array],
        features: Optional[T_Batch_Features_Array]
    ):
        return x, y, features

    def inverse_transform(
        self,
        x: Optional[T_Batch_Token_Array],
        y: Optional[T_Batch_Label_Array],
        features: Optional[T_Batch_Features_Array]
    ):
        return x, y, features


T_DatasetTransformerFactory = Callable[[], DatasetTransformer]
