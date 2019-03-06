import re
from typing import List

import numpy as np
from src.utils.Label import Label


class LASLabel(Label):
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 3
    SOS_TOKEN = '<sos>'
    SOS_INDEX = 1
    EOS_TOKEN = '<eos>'
    EOS_INDEX = 2
    UNK_TOKEN = '<unk>'
    UNK_INDEX = 0

    FIRST_INDEX = ord('a') - 1 - SPACE_INDEX  # 0 is reserved to space

    # a-z (26), space (1) , eos (1), sos (1), pad (1) -> 30
    num_classes = ord('z') - ord('a') + 5

    def __init__(self, transcription: str):
        transcription = re.sub('[;:!@#$?.,_\'\"\-]', '', transcription)

        self.__text: str = transcription
        self.__targets = transcription.replace(' ', '  ').split(' ')
        self.__indices = None
        self.__indices = self.to_index()
        if True in (self.__indices < 0):
            print('Character not supported')

    @property
    def transcription(self) -> str:
        return self.__text

    @property
    def character_list(self) -> List[str]:
        return list(self.transcription)

    @property
    def word_list(self) -> List[str]:
        return self.__targets

    def to_index(self) -> np.ndarray:
        if self.__indices is None:
            # Adding blank label
            index = list(np.hstack([self.SPACE_TOKEN if x == '' else list(x) for x in self.__targets]))
            # index = [self.SOS_TOKEN] + index + [self.EOS_TOKEN]
            # Transform char into index
            index_list = []
            for x in index:
                if x == self.SPACE_TOKEN:
                    index_list.append(self.SPACE_INDEX)
                # elif x == self.SOS_TOKEN:
                #     index_list.append(self.SOS_INDEX)
                # elif x == self.EOS_TOKEN:
                #     index_list.append(self.EOS_INDEX)
                else:
                    index_list.append(ord(x) - self.FIRST_INDEX)

            index = np.asarray(index_list)

            return index
        else:
            return self.__indices

    @staticmethod
    def from_index(seq) -> str:
        char_list = []
        for x in seq:
            if x == LASLabel.SPACE_INDEX:
                char_list.append(' ')
            elif x == LASLabel.SOS_INDEX or x == LASLabel.EOS_INDEX or x == LASLabel.UNK_INDEX:
                continue
            else:
                char_list.append(chr(x + ord('a') - LASLabel.UNK_INDEX - 1))
        str_decoded = ''.join(char_list)
        # # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')

        str_decoded = list(filter(None, re.split("([A-Z][^A-Z]*)", str_decoded)))
        str_decoded = [s.lower() for s in str_decoded]
        str_decoded = ' '.join(str_decoded)

        return str_decoded

    def __str__(self):
        return str(self.__indices)
