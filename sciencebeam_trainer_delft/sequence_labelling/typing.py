from typing import Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt


T_Batch_Token_Array = npt.NDArray[np.object_]       # shape: (batch, seq_len)
T_Batch_Label_Array = npt.NDArray[np.object_]       # shape: (batch, seq_len)
T_Batch_Features_Array = npt.NDArray[np.float32]  # shape: (batch, seq_len, feat_dim)

T_Document_Token_List = Sequence[str]  # shape: (seq_len)
T_Batch_Token_List = Sequence[T_Document_Token_List]  # shape: (batch, seq_len)

T_Document_Label_List = Sequence[str]  # shape: (seq_len)
T_Batch_Label_List = Sequence[T_Document_Label_List]  # shape: (batch, seq_len)

T_Document_Features_List = Sequence[Sequence[str]]  # shape: (seq_len, feat_dim)
T_Batch_Features_List = Sequence[T_Document_Features_List]  # shape: (batch, seq_len, feat_dim)

# Note: the 'Or_List' types are used to indicate that either numpy arrays or lists are accepted
# (better avoid if possible due to less precise typing)
T_Batch_Token_Array_Or_List = Union[T_Batch_Token_Array, T_Batch_Token_List]
T_Batch_Label_Array_Or_List = Union[T_Batch_Label_Array, T_Batch_Label_List]
T_Batch_Features_Array_Or_List = Union[T_Batch_Features_Array, T_Batch_Features_List]

T_Batch_Token_Type_Var = TypeVar(
    'T_Batch_Token_Type_Var',
    bound=T_Batch_Token_Array_Or_List
)
T_Batch_Label_Type_Var = TypeVar(
    'T_Batch_Label_Type_Var',
    bound=T_Batch_Label_Array_Or_List
)
T_Batch_Features_Type_Var = TypeVar(
    'T_Batch_Features_Type_Var',
    bound=T_Batch_Features_Array_Or_List
)

T_Optional_Batch_Token_Type_Var = TypeVar(
    'T_Optional_Batch_Token_Type_Var',
    bound=Optional[T_Batch_Token_Array_Or_List]
)
T_Optional_Batch_Label_Type_Var = TypeVar(
    'T_Optional_Batch_Label_Type_Var',
    bound=Optional[T_Batch_Label_Array_Or_List]
)
T_Optional_Batch_Features_Type_Var = TypeVar(
    'T_Optional_Batch_Features_Type_Var',
    bound=Optional[T_Batch_Features_Array_Or_List]
)
