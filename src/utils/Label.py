from typing import List

import numpy as np


class Label:

    def __init__(self, transcription: str):
        raise NotImplemented("transcription() not implemented")

    @property
    def transcription(self) -> str:
        raise NotImplemented("transcription() not implemented")

    @property
    def character_list(self) -> List[str]:
        raise NotImplemented("character_list() not implemented")

    @property
    def word_list(self) -> List[str]:
        raise NotImplemented("word_list() not implemented")

    def to_index(self) -> np.ndarray:
        raise NotImplemented("to_index() not implemented")

    @staticmethod
    def from_index(seq) -> str:
        raise NotImplemented("from_index() not implemented")

    def __str__(self):
        raise NotImplemented("str() not implemented")
