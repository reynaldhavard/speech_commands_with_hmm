import os


def generate_training_list(
    data_folder: str,
    output_list_file: str,
    validation_list_file: str,
    testing_list_file: str,
):
    validation_txt = os.path.join(data_folder, validation_list_file)
    testing_txt = os.path.join(data_folder, testing_list_file)

    with open(validation_txt) as f:
        validation_files = f.read().split("\n")[:-1]
    with open(testing_txt) as f:
        testing_files = f.read().split("\n")[:-1]

    all_audio_files = []
    for dirpath, dirnames, filenames in os.walk(data_folder):
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

    with open(os.path.join(data_folder, output_list_file), "w") as f:
        for filename in training_files:
            f.write(f"{filename}\n")
