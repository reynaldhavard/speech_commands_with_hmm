from speech_command_detection.utils import load_parquet_file
from speech_command_detection.hmm import HMM

import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    N_COMPONENTS = 5
    N_MIXTURES = 4
    N_ITER = 10
    LEFT_RIGHT = True
    N_MFCC = 13
    USE_DELTA = True
    USE_DELTA2 = True
    LABELS_FILE = "labels.txt"
    TRAINING_FEATURES = "./data/training_features.parquet"
    MODELS_FOLDER = "models"
    MODEL_FILE = "HMMModels.pkl"

    with open(os.path.join(CURRENT_DIR, LABELS_FILE)) as f:
        labels = f.read().split("\n")

    training_df = load_parquet_file(os.path.join(CURRENT_DIR, TRAINING_FEATURES))

    hmms = HMM(
        n_components=N_COMPONENTS,
        n_mixtures=N_MIXTURES,
        n_iter=N_ITER,
        labels=labels,
        left_right=LEFT_RIGHT,
        n_mfcc=N_MFCC,
        use_delta=USE_DELTA,
        use_delta2=USE_DELTA2,
    )

    hmms.fit(training_df)

    try:
        os.makedirs(os.path.join(CURRENT_DIR, MODELS_FOLDER))
    except:
        pass
    hmms.save(os.path.join(CURRENT_DIR, MODELS_FOLDER, MODEL_FILE))
