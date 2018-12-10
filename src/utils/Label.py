import numpy as np


class Label:

    @property
    def transcription(self) -> str:
        raise NotImplemented("transcription() not implemented")

    def to_index(self) -> np.ndarray:
        raise NotImplemented("to_index() not implemented")

    @staticmethod
    def from_index(seq) -> str:
        raise NotImplemented("from_index() not implemented")

    def __str__(self):
        raise NotImplemented("str() not implemented")
