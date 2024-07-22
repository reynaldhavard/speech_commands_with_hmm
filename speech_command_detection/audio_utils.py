import librosa
import numpy as np

from typing import Tuple


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(filename)

    return y, sr


def get_mfcc(
    raw_audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    compute_delta: bool = True,
    compute_delta2: bool = True,
) -> np.ndarray:
    if compute_delta2 and not compute_delta:
        raise Exception("compute_delta needs to be set to True to use compute_delta2")

    mfcc = librosa.feature.mfcc(y=raw_audio, sr=sample_rate, n_mfcc=n_mfcc)
    features = [mfcc]

    if compute_delta:
        mfcc_delta = librosa.feature.delta(mfcc)
        features.append(mfcc_delta)

    if compute_delta2:
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(mfcc_delta2)

    features = np.concatenate(features, axis=0).T

    return features


def normalize_mfcc(x: np.ndarray) -> np.ndarray:
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)

    return x
