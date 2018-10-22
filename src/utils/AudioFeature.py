import numpy as np
import python_speech_features as features
import scipy.io.wavfile as wav
from scipy import signal
from typing import Tuple


class FeatureConfig:
    def __init__(self):
        self.feature_type: str = 'mfcc'     # 'spec', 'log_spec'
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
        self.__feature = np.empty(0)
        self.__fs: float = None

    def __len__(self):
        return len(self.__feature)

    @property
    def feature(self) -> np.ndarray:
        return self.__feature

    @property
    def fs(self) -> float:
        return self.__fs

    def mfcc(self, audio, fs: float, winlen: float, winstep: float, numcep: int,
             nfilt: int, nfft: int, lowfreq, highfreq, preemph: float) -> np.ndarray:

        return features.mfcc(audio, samplerate=fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

    def specgram(self, audio, fs, nfft=1024, window_size=20, step_size=10, eps=1e-10):
        nperseg = int(round(window_size * fs / 1e3))
        noverlap = int(round(step_size * fs / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                                fs=fs,
                                                window='hann',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False,
                                                nfft=nfft)
        return freqs, times, spec.T.astype(np.float32)

    def log_specgram(self, audio, fs, nfft=1024, window_size=20,
                     step_size=10, eps=1e-10) -> Tuple[np.ndarray, np.ndarray, None]:

        freqs, times, spec = self.specgram(audio, fs, nfft, window_size, step_size, eps)

        return freqs, times, np.log(spec + eps)

    @staticmethod
    def fromFile(wav_name: str,
                 feature_config: FeatureConfig,
                 normalize_audio=True) -> 'AudioFeature':
        # Read the wav file
        fs, signal = wav.read(wav_name)
        return AudioFeature.fromAudio(signal, fs, feature_config, normalize_audio)

    @staticmethod
    def fromFeature(feature: Tuple[np.ndarray, np.ndarray, None], fs: float=None) -> 'AudioFeature':
        audio_feature = AudioFeature()
        audio_feature.__feature = feature
        audio_feature.__fs = fs

        return audio_feature

    @staticmethod
    def fromAudio(audio: np.ndarray, fs: float, feature_config: FeatureConfig, normalize_audio=True) -> 'AudioFeature':

        if normalize_audio:
            audio = audio / abs(max(audio))
        feature = AudioFeature()
        feature.__fs = fs
        if feature_config.feature_type is 'spec':
            _, _, feature.__feature = feature.specgram(
                audio=audio,
                fs=fs,
                nfft=feature_config.nfft,
                window_size=feature_config.winlen,
                step_size=feature_config.winstride)
        elif feature_config.feature_type is 'log_spec':
            _, _, feature.__feature = feature.log_specgram(
                audio=audio,
                fs=fs,
                nfft=feature_config.nfft,
                window_size=feature_config.winlen,
                step_size=feature_config.winstride)
        elif feature_config.feature_type is 'mfcc':
            if feature_config.highfreq is None:
                feature_config.highfreq = fs/2
            feature.__feature = feature.mfcc(
                audio=audio,
                fs=fs,
                numcep=feature_config.num_ceps,
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
