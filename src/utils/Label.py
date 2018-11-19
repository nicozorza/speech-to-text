import re
import numpy as np


class Label:
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    ENIE_TOKEN = '<enie>'
    ENIE_INDEX = ord('Z') - ord('A') + ord('z')-ord('a') + 1 + 1
    FIRST_INDEX = ord('A')

    def __init__(self, transcription: str):
        transcription = re.sub('[;:!@#$?.,_\'\"\-]', '', transcription)

        transcription = transcription.lower()
        self.__text: str = transcription
        self.__targets = transcription.split(' ')
        self.__targets = [w.capitalize() for w in self.__targets]
        self.__indices = None
        self.__indices = self.to_index()
        if True in (self.__indices < 0):
            print('Character not supported')

    @property
    def transcription(self) -> str:
        return self.__text

    def to_index(self) -> np.ndarray:
        if self.__indices is None:
            # Adding blank label
            index = np.hstack([list(x) for x in self.__targets])
            # Transform char into index
            index_list = []
            for x in index:
                if ord('A') <= ord(x) <= ord('Z'):
                    index_list.append(ord(x) - self.FIRST_INDEX)

                else:
                    if x == 'ñ':
                        index_list.append(self.ENIE_INDEX)
                    else:
                        index_list.append(ord(x) - self.FIRST_INDEX - (ord('a') - ord('Z')) + 1)

            index = np.asarray(index_list)

            return index
        else:
            return self.__indices

    @staticmethod
    def from_index(seq) -> str:
        char_list = []
        for x in seq:
            if 0 <= x <= (ord('Z')-ord('A')):
                char_list.append(chr(x+ord('A')))

            elif x == Label.ENIE_INDEX:  # Replace enie
                char_list.append(chr(241))
            else:
                char_list.append(chr(x + ord('a') - (ord('Z') - ord('A')) - 1))
        str_decoded = ''.join(char_list)
        # # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')

        str_decoded = list(filter(None, re.split("([A-Z][^A-Z]*)", str_decoded)))
        str_decoded = [s.lower() for s in str_decoded]
        str_decoded = ' '.join(str_decoded)

        return str_decoded

    def __str__(self):
        return str(self.__indices)
