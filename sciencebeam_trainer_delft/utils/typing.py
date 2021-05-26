from typing import TypeVar

from typing_extensions import Protocol


T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


class GetSetStateProtocol(Protocol):
    def __getstate__(self) -> dict:
        pass

    def __setstate__(self, state: dict):
        pass


T_GetSetStateProtocol = TypeVar('T_GetSetStateProtocol', bound=GetSetStateProtocol)
