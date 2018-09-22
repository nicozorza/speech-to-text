import numpy as np
import python_speech_features as features
import scipy.io.wavfile as wav
from scipy import signal
from typing import Tuple


class FeatureConfig:
    def __init__(self):
        self.feature_type: str = 'spec'     # 'mfcc'
        self.nfft: int = 1024
        self.winlen: int = 20
        self.winstride: int = 10

        # MFCC parameters
        self.num_ceps: int = 13
        self.num_filters: int = 26
        self.lowfreq: float = 0
        self.highfreq: float = None
        self.preemph: float = 0.98


class AudioFeature:
    def __init__(self):
        self.__audio = np.empty(0)
        self.__feature = np.empty(0)
        self.__fs = 0

    def __len__(self):
        return len(self.__feature)

    def getSamplingRate(self) -> float:
        return self.__fs

    def getFeature(self) -> Tuple[np.ndarray, np.ndarray, None]:
        return self.__feature

    def getAudio(self) -> np.ndarray:
        return self.__audio

    def mfcc(self,
                winlen: float,
                winstep: float,
                numcep: int,
                nfilt: int,
                nfft: int,
                lowfreq,
                highfreq,
                preemph: float) -> np.ndarray:

        return features.mfcc(self.__audio, samplerate=self.__fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def log_specgram(self, nfft=1024, window_size=20, step_size=10, eps=1e-10) -> Tuple[np.ndarray, np.ndarray, None]:
        nperseg = int(round(window_size * self.__fs / 1e3))
        noverlap = int(round(step_size * self.__fs / 1e3))
        freqs, times, spec = signal.spectrogram(self.__audio,
                                                fs=self.__fs,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False,
                                                nfft=nfft)
        return freqs, times, np.log(spec.T.astype(np.float32) + eps)

    @staticmethod
    def fromFile(wav_name: str,
                 feature_config: FeatureConfig,
                 normalize_audio=True) -> 'AudioFeature':
        # Read the wav file
        fs, signal = wav.read(wav_name)
        return AudioFeature.fromAudio(signal, fs, feature_config, normalize_audio)

    @staticmethod
    def fromFeature(feature: Tuple[np.ndarray, np.ndarray, None], nfft: int) -> 'AudioFeature':
        audio_feature = AudioFeature()
        audio_feature.__feature = feature

        return audio_feature

    @staticmethod
    def fromAudio(audio: np.ndarray, fs: float, feature_config: FeatureConfig,
                     normalize_audio=True) -> 'AudioFeature':

        if normalize_audio:
            audio = audio / abs(max(audio))
        feature = AudioFeature()
        feature.__audio = audio
        feature.__fs = fs
        if feature_config.feature_type is 'spec':
            _, _, feature.__feature = feature.log_specgram(nfft=feature_config.nfft,
                                                           window_size=feature_config.winlen,
                                                           step_size=feature_config.winstride)
        elif feature_config.feature_type is 'mfcc':
            if feature_config.highfreq is None:
                feature_config.highfreq = fs/2
            feature.__feature = feature.mfcc(numcep=feature_config.num_ceps,
                                             winlen=feature_config.winlen/1000,
                                             winstep=feature_config.winstride/1000,
                                             nfft=feature_config.nfft,
                                             lowfreq=feature_config.lowfreq,
                                             highfreq=feature_config.highfreq,
                                             preemph=feature_config.preemph,
                                             nfilt=feature_config.num_filters)
        else:
            raise ValueError("Wrong feature type. Only MFCC and spectogram are available.")

        return feature
