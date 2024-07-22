from speech_command_detection.audio_utils import *

import pandas as pd

import os
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_training_list(
    data_folder: str,
    output_list_file: str,
    validation_list_file: str,
    testing_list_file: str,
) -> None:
    validation_txt = os.path.join(CURRENT_DIR, data_folder, validation_list_file)
    testing_txt = os.path.join(CURRENT_DIR, data_folder, testing_list_file)

    with open(validation_txt) as f:
        validation_files = f.read().split("\n")[:-1]
    with open(testing_txt) as f:
        testing_files = f.read().split("\n")[:-1]

    all_audio_files = []
    for dirpath, _, filenames in os.walk(os.path.join(CURRENT_DIR, data_folder)):
        for f in filenames:
            full_filename = os.path.join(dirpath, f)
            if (
                full_filename.endswith(".wav")
                and "_background_noise_" not in full_filename
            ):
                all_audio_files.append("/".join(full_filename.split("/")[-2:]))

    training_files = sorted(
        list(set(all_audio_files) - set(validation_files) - set(testing_files))
    )

    with open(os.path.join(CURRENT_DIR, data_folder, output_list_file), "w") as f:
        for filename in training_files:
            f.write(f"{filename}\n")


def generate_labels(example_file: str, output_file: str) -> None:
    with open(os.path.join(CURRENT_DIR, example_file)) as f:
        file_list = f.read().split("\n")[:-1]

    labels = sorted(list(set([file.split("/")[0] for file in file_list])))

    with open(os.path.join(CURRENT_DIR, output_file), "w") as f:
        f.write("\n".join(labels))


def generate_parquet_file(
    data_list_file: str,
    data_folder: str,
    output_file: str,
    n_mfcc: int = 13,
    compute_delta: bool = False,
    compute_delta2: bool = False,
) -> None:
    with open(os.path.join(CURRENT_DIR, data_folder, data_list_file)) as f:
        data_list = f.read().split("\n")[:-1]

    feature_dict = dict()
    for filename in tqdm(data_list):
        full_filename = os.path.join(CURRENT_DIR, data_folder, filename)
        label = filename.split("/")[0]
        raw_audio, sr = load_audio(full_filename)
        mfcc = get_mfcc(
            raw_audio=raw_audio,
            sample_rate=sr,
            n_mfcc=n_mfcc,
            compute_delta=compute_delta,
            compute_delta2=compute_delta2,
        )
        normalized_mfcc = normalize_mfcc(mfcc)
        feature_dict[filename] = {
            "label": label,
            "normalized_mfcc": normalized_mfcc.tolist(),
        }

    df = pd.DataFrame.from_dict(feature_dict, orient="index")

    full_output_filename = os.path.join(CURRENT_DIR, data_folder, output_file)
    df.to_parquet(full_output_filename)
    print(f"Saved processed data to {full_output_filename}")


if __name__ == "__main__":
    DATA_FOLDER = "../data"
    TRAINING_LIST = "training_list.txt"
    VALIDATION_LIST = "validation_list.txt"
    TESTING_LIST = "testing_list.txt"
    TRAINING_FEATURES = "training_features.parquet"
    VALIDATION_FEATURES = "validation_features.parquet"
    TESTING_FEATURES = "testing_features.parquet"
    LABELS_FILE = "../labels.txt"

    generate_training_list(
        data_folder=DATA_FOLDER,
        output_list_file=TRAINING_LIST,
        validation_list_file=VALIDATION_LIST,
        testing_list_file=TESTING_LIST,
    )

    generate_labels(
        example_file=os.path.join(DATA_FOLDER, VALIDATION_LIST),
        output_file=LABELS_FILE,
    )

    generate_parquet_file(
        data_list_file=TRAINING_LIST,
        data_folder=DATA_FOLDER,
        output_file=TRAINING_FEATURES,
        n_mfcc=13,
        compute_delta=True,
        compute_delta2=True,
    )
    generate_parquet_file(
        data_list_file=VALIDATION_LIST,
        data_folder=DATA_FOLDER,
        output_file=VALIDATION_FEATURES,
        n_mfcc=13,
        compute_delta=True,
        compute_delta2=True,
    )
    generate_parquet_file(
        data_list_file=TESTING_LIST,
        data_folder=DATA_FOLDER,
        output_file=TESTING_FEATURES,
        n_mfcc=13,
        compute_delta=True,
        compute_delta2=True,
    )
