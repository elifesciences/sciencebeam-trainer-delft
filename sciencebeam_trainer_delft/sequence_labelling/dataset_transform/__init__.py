from abc import ABC, abstractmethod
from typing import Callable, Tuple

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Label_Type_Var,
    T_Batch_Token_Type_Var,
    T_Optional_Batch_Features_Type_Var,
    T_Optional_Batch_Label_Type_Var,
    T_Optional_Batch_Token_Type_Var
)


class DatasetTransformer(ABC):
    @abstractmethod
    def fit_transform(
        self,
        x: T_Batch_Token_Type_Var,
        y: T_Optional_Batch_Label_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ) -> Tuple[
        T_Batch_Token_Type_Var,
        T_Optional_Batch_Label_Type_Var,
        T_Optional_Batch_Features_Type_Var
    ]:
        pass

    @abstractmethod
    def inverse_transform(
        self,
        x: T_Optional_Batch_Token_Type_Var,
        y: T_Optional_Batch_Label_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ) -> Tuple[
        T_Optional_Batch_Token_Type_Var,
        T_Optional_Batch_Label_Type_Var,
        T_Optional_Batch_Features_Type_Var
    ]:
        pass

    def fit_transform_x_and_features(
        self,
        x: T_Batch_Token_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ):
        x, _, features = self.fit_transform(x, None, features)
        return x, features

    def inverse_transform_y(
        self,
        y: T_Batch_Label_Type_Var,
    ) -> T_Batch_Label_Type_Var:
        _, inverse_transformed_y, _ = self.inverse_transform(None, y, None)
        assert inverse_transformed_y is not None
        return inverse_transformed_y


class DummyDatasetTransformer(DatasetTransformer):
    def fit_transform(
        self,
        x: T_Batch_Token_Type_Var,
        y: T_Optional_Batch_Label_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ) -> Tuple[
        T_Batch_Token_Type_Var,
        T_Optional_Batch_Label_Type_Var,
        T_Optional_Batch_Features_Type_Var
    ]:
        return x, y, features

    def inverse_transform(
        self,
        x: T_Optional_Batch_Token_Type_Var,
        y: T_Optional_Batch_Label_Type_Var,
        features: T_Optional_Batch_Features_Type_Var
    ) -> Tuple[
        T_Optional_Batch_Token_Type_Var,
        T_Optional_Batch_Label_Type_Var,
        T_Optional_Batch_Features_Type_Var
    ]:
        return x, y, features


T_DatasetTransformerFactory = Callable[[], DatasetTransformer]
