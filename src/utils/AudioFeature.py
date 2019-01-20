import numpy as np
import python_speech_features as features
import scipy.io.wavfile as wav
from scipy import signal
from typing import Tuple


class FeatureConfig:
    def __init__(self):
        self.feature_type: str = 'mfcc'     # 'spec', 'log_spec', 'deep_speech_mfcc'
        self.nfft: int = 1024
        self.winlen: int = 20
        self.winstride: int = 10

        # MFCC parameters
        self.num_ceps: int = 13
        self.num_filters: int = 26
        self.lowfreq: float = 0
        self.highfreq: float = None
        self.preemph: float = 0.98

        self.mfcc_window = np.hamming
        # Las posibles ventanas para los MFCC son:
        # np.hamming, np.hanning, np.kaiser, np.blackman, np.bartlett

        self.spec_window: str = 'hann'
        # Las posibles ventanas para el espectograma son:
        # `boxcar`, `triang`, `blackman`, `hamming`, `hann`, `bartlett`,
        # `flattop`, `parzen`, `bohman`, `blackmanharris`, `nuttall`,
        # `barthann`, `kaiser` (needs beta), `gaussian` (needs standard
        # deviation), `general_gaussian` (needs power, width), `slepian`
        # (needs width), `dpss` (needs normalized half-bandwidth),
        # `chebwin` (needs attenuation), `exponential` (needs decay scale),
        # `tukey` (needs taper fraction)

        self.num_context: int = 9   # Cantidad de features de contexto (ver DeepSpeech)


class AudioFeature:
    def __init__(self):
        self.__feature = np.empty(0)
        self.__fs: float = None
        self.num_features: int = None

    def __len__(self):
        return len(self.__feature)

    @property
    def feature(self) -> np.ndarray:
        return self.__feature

    @property
    def fs(self) -> float:
        return self.__fs

    def mfcc(self, audio, fs: float, winlen: float, winstep: float, numcep: int,
             nfilt: int, nfft: int, lowfreq, highfreq, preemph: float, winfunc) -> np.ndarray:

        return features.mfcc(audio, samplerate=fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
                             preemph=preemph, winfunc=winfunc)

    def deep_speech_mfcc(self, audio, fs: float, winlen: float, winstep: float, numcep: int,
                         nfilt: int, nfft: int, lowfreq, highfreq, preemph: float, winfunc,
                         num_context: int) -> np.ndarray:

        features = self.mfcc(audio, fs=fs, winlen=winlen, winstep=winstep, numcep=numcep,
                             nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq,
                             preemph=preemph, winfunc=winfunc)

        # We only keep every second feature (BiRNN stride = 2)
        features = features[::2]

        num_strides = len(features)

        empty_context = np.zeros((num_context, numcep), dtype=features.dtype)

        features = np.concatenate((empty_context, features, empty_context))

        window_size = 2 * num_context + 1
        strided_features = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        strided_features = np.reshape(strided_features, [num_strides, -1])

        # Copy the strided array so that we can write to it safely
        strided_features = np.copy(strided_features)
        strided_features = (strided_features - np.mean(strided_features)) / np.std(strided_features)

        return strided_features

    def specgram(self, audio, fs, nfft=1024, window_size=20, step_size=10, eps=1e-10, window: str = 'hann'):
        nperseg = int(round(window_size * fs / 1e3))
        noverlap = int(round(step_size * fs / 1e3))
        freqs, times, spec = signal.spectrogram(audio,
                                                fs=fs,
                                                window=window,
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                detrend=False,
                                                nfft=nfft)
        return freqs, times, spec.T.astype(np.float32)

    def log_specgram(self, audio, fs, nfft=1024, window_size=20,
                     step_size=10, eps=1e-10, window='hann') -> Tuple[np.ndarray, np.ndarray, None]:

        freqs, times, spec = self.specgram(audio, fs, nfft, window_size, step_size, eps, window)

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
            feature.num_features = feature_config.nfft
            _, _, feature.__feature = feature.specgram(
                audio=audio,
                fs=fs,
                nfft=feature_config.nfft,
                window_size=feature_config.winlen,
                step_size=feature_config.winstride)
        elif feature_config.feature_type is 'log_spec':
            feature.num_features = feature_config.nfft
            _, _, feature.__feature = feature.log_specgram(
                audio=audio,
                fs=fs,
                nfft=feature_config.nfft,
                window_size=feature_config.winlen,
                step_size=feature_config.winstride,
                window=feature_config.spec_window
            )
        elif feature_config.feature_type is 'mfcc':
            feature.num_features = feature_config.num_ceps
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
                nfilt=feature_config.num_filters,
                winfunc=feature_config.mfcc_window)

        elif feature_config.feature_type is 'deep_speech_mfcc':
            feature.num_features = feature_config.num_ceps
            if feature_config.highfreq is None:
                feature_config.highfreq = fs / 2
            feature.__feature = feature.deep_speech_mfcc(
                audio=audio,
                fs=fs,
                numcep=feature_config.num_ceps,
                winlen=feature_config.winlen / 1000,
                winstep=feature_config.winstride / 1000,
                nfft=feature_config.nfft,
                lowfreq=feature_config.lowfreq,
                highfreq=feature_config.highfreq,
                preemph=feature_config.preemph,
                nfilt=feature_config.num_filters,
                winfunc=feature_config.mfcc_window,
                num_context=feature_config.num_context
            )

        else:
            raise ValueError("Wrong feature type. Only MFCC and spectogram are available.")

        return feature
