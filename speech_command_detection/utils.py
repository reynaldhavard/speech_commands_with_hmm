import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import List, Tuple


def load_parquet_file(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    return df


def get_data_per_label(
    df: pd.DataFrame,
    label: str = None,
    n_mfcc: int = 13,
    use_delta: bool = True,
    use_delta2: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List]:
    if use_delta2 and not use_delta:
        raise Exception("use_delta needs to be set to True to use use_delta2")

    if label is None:
        filtered_df = df.copy()
    else:
        filtered_df = df[df["label"] == label]
    num_features = n_mfcc
    if use_delta:
        num_features = n_mfcc * 2
    if use_delta2:
        num_features = n_mfcc * 3

    arrays = (
        filtered_df["normalized_mfcc"]
        .apply(lambda x: np.vstack(x)[:num_features])
        .tolist()
    )
    full_array = np.concatenate(arrays, axis=0)
    column_counts = np.array([arr.shape[0] for arr in arrays])
    labels = filtered_df["label"].tolist()

    return full_array, column_counts, labels


def plot_confusion_matrix(conf_matrix_df: pd.DataFrame):
    _, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        conf_matrix_df,
        annot=True,
        fmt="d",
        ax=ax,
        annot_kws={"size": 9},
        cbar=False,
        cmap="viridis",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Groundtruth")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labelrotation=90)
    plt.show()
