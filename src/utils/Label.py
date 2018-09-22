import re
import numpy as np


class Label:
    # Constants
    SPACE_TOKEN = '<space>'
    SPACE_INDEX = 0
    FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

    def __init__(self, transcription: str):
        transcription = re.sub('[;:!@#$?.,_\'\"\-]', '', transcription)

        self.__text: str = transcription
        # Delete blanks at the beginning and the end of the transcription, transform to lowercase,
        # delete numbers in the beginning, etc.
        self.__targets = (' '.join(transcription.strip().lower().split(' ')[2:]).replace('.', '')).replace(' ', '  ').split(' ')
        self.__indices = None
        self.__indices = self.toIndex()
        if True in (self.__indices < 0):
            print('Character not supported')

    def getTranscription(self) -> str:
        return self.__text

    def toIndex(self) -> np.ndarray:
        if self.__indices is None:
            # Adding blank label
            index = np.hstack([self.SPACE_TOKEN if x == '' else list(x) for x in self.__targets])
            # Transform char into index
            index = np.asarray([self.SPACE_INDEX if x == '<space>' else ord(x) - self.FIRST_INDEX for x in index])
            return index
        else:
            return self.__indices

    def __str__(self):
        return str(self.__indices)

    @staticmethod
    def fromFile(file_name: str):
        with open(file_name, 'r') as f:
            transcription = f.readlines()[0]  # This method assumes that the transcription is in the first line
            return Label(transcription)  # Create Label class from transcription