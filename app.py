from speech_command_detection.hmm import HMM

from audiorecorder import audiorecorder
import librosa
import numpy as np
import streamlit as st

import logging


logging.getLogger("hmmlearn").setLevel("CRITICAL")


def main(models_path: str, labels_path: str, sample_rate: int) -> None:
    hmms = HMM.load(models_path)

    with open(labels_path) as f:
        labels = f.read().split("\n")

    st.title("Speech Commands Demo")
    st.write(f"Pronounce one of the following words:")
    num_columns = 5
    cols = st.columns(num_columns)
    for i, label in enumerate(labels):
        with cols[i % num_columns]:
            st.write(label)

    audio = audiorecorder("Click to record", "Click to stop recording")

    if len(audio) > 0:
        audio_array = np.array(audio.get_array_of_samples())
        audio_array = audio_array.astype(np.float32, order="C") / 32768.0

        audio_array = librosa.resample(
            audio_array, orig_sr=audio.frame_rate, target_sr=sample_rate
        )
        st.audio(audio_array, sample_rate=sample_rate)

        with st.spinner("Predicting..."):
            predicted_label = hmms.predict(
                raw_audio=audio_array, sample_rate=sample_rate
            )

        st.write(f"Prediction: {predicted_label}")


if __name__ == "__main__":
    MODELS_PATH = "models/HMMModels.pkl"
    LABELS_PATH = "labels.txt"
    SAMPLE_RATE = 22050

    main(MODELS_PATH, LABELS_PATH, SAMPLE_RATE)
