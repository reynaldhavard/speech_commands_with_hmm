# Speech Commands with Hidden Markov Models and Gaussian Mixture Models

This repository implements an approach to solving the problem of keyword spotting.

It fits Hidden Markov Models (HMM) with Gaussian Mixture emissions on the [Speech Commands Dataset](https://research.google/blog/launching-the-speech-commands-dataset/) (v0.0.1), which is a dataset containing multiple ~1 second files for 30 different words and background noises.

While one of the goals from the dataset was to help build models to detect specifically 10 words among the 30 and consider the other ones as "unknown", plus detecting silence, this repository provides code to fit multiple HMMs (one per word) in order to discriminate among these 30 words. 

It also provides a way to launch an app to test the model with your own voice.

## How does it work?

- The dataset contains ~1 audio files for 30 different word categories.
- Each file will have some features extracted from it, specifically MFCCs with delta and delta-delta features. These features are normalized.
- For each word, we initialize a HMM. Each HMM has the same number of states/components and the same number of Gaussian mixtures to represent the emission distributions. It is possible to choose whether we want our HMM to be left-right or ergodic, and whether we want to use the delta and delta-delta features.
- Each HMM is fitted to audio files corresponding to a specific word, e.g. the HMM for the word "bed" will be fitted using features from audio files containing the word "bed" and no other word.


## How to use this repo?

- First, create a virtual environment (for example with `virtualenv`), activate this environment and run:
```
pip install -r requirements.txt
pip install -e .
```
- Download the data executing `./get_data.sh`
- Running `speech_commands_detection/prepare_data.py` generates a list of training files (not provided by the original dataset, so this will be all the audio files without the validation and test sets), generates a list of the 30 labels and generates parquet files containing the audio features (MFCC, delta and delta-delta) for each file, for each set.
- Run `train.py` to train the models.

You can have a look at the `confusion_matrix.ipynb` notebook to check the performance of a specific model on the validation set or the testing set via a confusion matrix.


## Test with your own voice

Whether you want to try the provided model or re-train your own model with different parameters, you can launch a Streamlit app and try with your own voice! Simply run `streamlit run app.py` (you do need to run the "pip install" commands first though) and it should start an app accessible via your browser.


## TODO

- Create external config files, instead of having config variables defined in the scripts.
- Add more metrics for evaluation.