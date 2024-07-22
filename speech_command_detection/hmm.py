from speech_command_detection.audio_utils import get_mfcc, normalize_mfcc
from speech_command_detection.utils import get_data_per_label

from hmmlearn import hmm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import logging
import pickle
from typing import List, Tuple
import warnings

logging.getLogger("hmmlearn").setLevel("CRITICAL")
warnings.filterwarnings("ignore")


class HMM:
    def __init__(
        self: "HMM",
        n_components: int,
        n_mixtures: int,
        n_iter: int,
        labels: List,
        left_right: bool = True,
        n_mfcc: int = 13,
        use_delta: bool = True,
        use_delta2: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_mixtures = n_mixtures
        self.left_right = left_right
        self.labels = sorted(labels)
        self.n_mfcc = n_mfcc
        self.use_delta = use_delta
        self.use_delta2 = use_delta2

        hmm_init_params, hmm_params = self._get_updated_training_params()
        self.models = {
            label: hmm.GMMHMM(
                n_components=n_components,
                n_mix=n_mixtures,
                n_iter=n_iter,
                init_params=hmm_init_params,
                params=hmm_params,
            )
            for label in labels
        }

        if self.left_right:
            self._init_prob_matrices_models()

        self.index_to_label = {i: label for i, label in enumerate(self.labels)}
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}

    def _get_updated_training_params(self: "HMM") -> Tuple[str, str]:
        if self.left_right:
            init_params = "mcw"
            params = "tmcw"
        else:
            init_params = "stmcw"
            params = "stmcw"

        return init_params, params

    def _init_prob_matrices_models(self: "HMM") -> None:
        startprob_init = self._get_startprob_matrix()
        transmat_init = self._get_transmat_matrix()
        for label in self.labels:
            self.models[label].startprob_ = startprob_init
            self.models[label].transmat_ = transmat_init

    def _get_startprob_matrix(self: "HMM") -> np.ndarray:
        startprob = np.zeros((self.n_components,))
        startprob[0] = 1

        return startprob

    def _get_transmat_matrix(self: "HMM") -> np.ndarray:
        transmat = np.zeros((self.n_components, self.n_components))
        for i in range(self.n_components - 1):
            transmat[i, i] = 0.5
            transmat[i, i + 1] = 0.5
        transmat[self.n_components - 1, self.n_components - 1] = 1

        return transmat

    def fit(self: "HMM", df: pd.DataFrame) -> None:
        labels_loop = tqdm(self.labels)
        for label in labels_loop:
            labels_loop.set_description(f'Processing label "{label}"')
            filtered_data, column_counts, _ = get_data_per_label(
                df=df,
                label=label,
                n_mfcc=self.n_mfcc,
                use_delta=self.use_delta,
                use_delta2=self.use_delta2,
            )
            self.models[label].fit(filtered_data, column_counts)

    def predict(self: "HMM", raw_audio: np.ndarray, sample_rate: int) -> str:
        mfcc = get_mfcc(
            raw_audio=raw_audio,
            sample_rate=sample_rate,
            n_mfcc=self.n_mfcc,
            compute_delta=self.use_delta,
            compute_delta2=self.use_delta2,
        )
        normalized_mfcc = normalize_mfcc(mfcc)

        scores = []
        for label in self.labels:
            score = self.models[label].score(normalized_mfcc)
            scores.append(score)

        prediction_index = np.argmax(scores)
        prediction_label = self.index_to_label[prediction_index]

        return prediction_label

    def evaluate(self: "HMM", df: pd.DataFrame) -> pd.DataFrame:
        data, column_counts, groundtruth_labels = get_data_per_label(
            df=df,
            label=None,
            n_mfcc=self.n_mfcc,
            use_delta=self.use_delta,
            use_delta2=self.use_delta2,
        )
        groundtruth_indexes = [
            self.label_to_index[label] for label in groundtruth_labels
        ]
        split_data = np.split(data, np.cumsum(column_counts))[:-1]
        scores = []
        labels_loop = tqdm(self.labels)
        for label in labels_loop:
            labels_loop.set_description(f'Evaluating data for label "{label}"')
            score_label = []
            for split in split_data:
                try:
                    score_label.append(self.models[label].score(split))
                except:
                    score_label.append(-np.inf)
            scores.append(np.array(score_label))
        scores = np.vstack(scores)
        prediction_indexes = np.argmax(scores, axis=0)

        conf_matrix = confusion_matrix(
            groundtruth_indexes,
            prediction_indexes,
            labels=list(range(len(self.labels))),
        )
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.columns = self.labels
        conf_matrix_df.index = self.labels

        return conf_matrix_df

    def save(self: "HMM", filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def load(filename: str) -> "HMM":
        with open(filename, "rb") as f:
            return pickle.load(f)
