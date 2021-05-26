from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from sciencebeam_trainer_delft.sequence_labelling.typing import (
    T_Batch_Tokens,
    T_Batch_Features,
    T_Batch_Labels
)


class DatasetTransformer(ABC):
    @abstractmethod
    def fit_transform(
        self,
        x: T_Batch_Tokens,
        y: Optional[T_Batch_Labels],
        features: Optional[T_Batch_Features]
    ) -> Tuple[T_Batch_Tokens, Optional[T_Batch_Labels], Optional[T_Batch_Features]]:
        pass

    @abstractmethod
    def inverse_transform(
        self,
        x: Optional[T_Batch_Tokens],
        y: Optional[T_Batch_Labels],
        features: Optional[T_Batch_Features]
    ) -> Tuple[Optional[T_Batch_Tokens], Optional[T_Batch_Labels], Optional[T_Batch_Features]]:
        pass

    def fit_transform_x_and_features(
        self,
        x: T_Batch_Tokens,
        features: Optional[T_Batch_Features]
    ):
        x, _, features = self.fit_transform(x, None, features)
        return x, features

    def inverse_transform_y(
        self,
        y: T_Batch_Labels,
    ) -> T_Batch_Labels:
        _, inverse_transformed_y, _ = self.inverse_transform(None, y, None)
        assert inverse_transformed_y is not None
        return inverse_transformed_y


class DummyDatasetTransformer(DatasetTransformer):
    def fit_transform(
        self,
        x: T_Batch_Tokens,
        y: Optional[T_Batch_Labels],
        features: Optional[T_Batch_Features]
    ):
        return x, y, features

    def inverse_transform(
        self,
        x: Optional[T_Batch_Tokens],
        y: Optional[T_Batch_Labels],
        features: Optional[T_Batch_Features]
    ):
        return x, y, features


T_DatasetTransformerFactory = Callable[[], DatasetTransformer]
