import os
import subprocess
from datetime import datetime

import gensim
import yaml
from gensim.models.word2vec import LineSentence
# from gensim.models.callbacks import CallbackAny2Vec


def get_env_var(var_name, cast_func=None, mandatory=False):
    var_content = os.getenv(var_name)
    if var_content is not None:
        print(f"{var_name}: {var_content.__repr__()}")
    elif mandatory:
        raise Exception(f"environment variable: '{var_name}' is mandatory")
    if cast_func:
        try:
            var_content = cast_func(var_content)
        except:
            raise Exception(f"Could not convert var '{var_name}' to {cast_func}")
    return var_content


def get_description():
    veld_file = None
    for file in os.listdir("/veld/input/"):
        if file.startswith("veld") and file.endswith("yaml"):
            if veld_file is not None:
                raise Exception("Multiple veld yaml files found.")
            else:
                veld_file = file
    if veld_file is None:
        print("no training data veld yaml file found. Won't be able to persist that as metadata.", flush=True)
    else:
        with open("/veld/input/" + veld_file, "r") as f:
            input_veld_metadata = yaml.safe_load(f)
            try:
                train_data_description = input_veld_metadata["x-veld"]["data"]["description"]
                return train_data_description
            except:
                pass


def train_and_persist(train_data_path, epochs, vector_size, window, min_count, cpu_count, out_model_path):

    # class LossLogger(CallbackAny2Vec):
    #     def __init__(self):
    #         self.epoch = 0
    # 
    #     def on_epoch_end(self, model):
    #         loss = model.get_latest_training_loss()
    #         print(f"epoch: {self.epoch}, loss: {loss}", flush=True)
    #         self.epoch += 1

    sentences = LineSentence(train_data_path)
    time_start = datetime.now()
    print("training start:", time_start, flush=True)
    model = gensim.models.Word2Vec(
        sentences=sentences,
        epochs=epochs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=cpu_count,
        # callbacks=[LossLogger()],
    )
    duration = (datetime.now() - time_start).seconds / 60
    # print(f"done. duration in minutes: {DURATION}", flush=True)
    time_end = datetime.now()
    print("training done:", time_end, flush=True)
    model.save(out_model_path)
    return duration


def write_metadata(
    duration, 
    train_data_path, 
    epochs, 
    vector_size, 
    window, 
    min_count, 
    cpu_count, 
    out_model_path, 
    model_description, 
    model_id, 
    training_architecture, 
    out_model_metadata_path,
    train_data_description,
):

    # calculate size of training and model data
    def calc_size(file_or_folder):
        size = subprocess.run(["du", "-sh", file_or_folder], capture_output=True, text=True)
        size = size.stdout.split()[0]
        return size
    train_data_size = calc_size(train_data_path)
    model_data_size = calc_size(out_model_path)

    # calculate hash of training data
    train_data_md5_hash = subprocess.run(["md5sum", train_data_path], capture_output=True, text=True)
    train_data_md5_hash = train_data_md5_hash.stdout.split()[0]

    # aggregate into metadata dictionary
    out_veld_metadata = {
        "x-veld": {
            "data": {
                "description": model_description,
                "file_type": "bin",
                "topics": [
                    "NLP",
                    "word embeddings",
                ],
                "contents": [
                    "word embeddings model",
                    "word2vec model",
                ],
                "additional": {
                    "model_id": model_id,
                    "training_architecture": training_architecture,
                    "train_data_description": train_data_description,
                    "train_data_size": train_data_size,
                    "train_data_md5_hash": train_data_md5_hash,
                    "training_vector_size": vector_size,
                    "training_epochs": epochs,
                    "window": window,
                    "min_count": min_count,
                    "training_duration (minutes)": round(duration, 1),
                    "model_data_size": model_data_size,
                }
            }
        }
    }

    # write to yaml
    with open(out_model_metadata_path, "w") as f:
        yaml.dump(out_veld_metadata, f, sort_keys=False)


def main():

    train_data_file = get_env_var("in_train_data_file")
    model_description = get_env_var("model_description")
    out_model_file = get_env_var("out_model_file")
    out_model_metadata_file = "/veld/output/veld.yaml"
    epochs = get_env_var("epochs", int)
    vector_size = get_env_var("vector_size", int)
    window = get_env_var("window", int)
    min_count = get_env_var("min_count", int)
    cpu_count = get_env_var("cpu_count", int)
    training_architecture = "word2vec"

    train_data_path_list = []
    out_model_path_list = []
    out_model_metadata_path_list = []
    if train_data_file and out_model_file:
        train_data_path_list.append("/veld/input/" + train_data_file)
        out_model_path_list.append("/veld/output/" + out_model_file)
        out_model_metadata_path_list.append("/veld/output/" + out_model_metadata_file)
    else:
        for file in os.listdir("/veld/input/"):
            file_name = "".join(file.split(".")[:-1])
            train_data_path_list.append("/veld/input/" + file)
            out_model_path_list.append("/veld/output/" + file_name + ".bin")
            out_model_metadata_path_list.append("/veld/output/" + file_name + ".yaml")

    for train_data_path, out_model_path, out_model_metadata_path in zip(train_data_path_list, out_model_path_list, out_model_metadata_path_list): 
        model_id = out_model_path.replace(".bin", "")

        print("train_data_path:", train_data_path)
        print("out_model_path:", out_model_path)
        print("out_model_metadata_path:", out_model_metadata_path)
        print("model_description:", model_description)
        print("model_id:", model_id)
        print("epochs:", epochs)
        print("vector_size:", vector_size)
        print("window:", window)
        print("min_count:", min_count)
        print("cpu_count:", cpu_count)
        print("training_architecture:", training_architecture)

        train_data_description = get_description()
        duration = train_and_persist(train_data_path, epochs, vector_size, window, min_count, cpu_count, out_model_path)
        write_metadata(
            duration, 
            train_data_path, 
            epochs, 
            vector_size, 
            window, 
            min_count, 
            cpu_count, 
            out_model_path, 
            model_description, 
            model_id, 
            training_architecture, 
            out_model_metadata_path,
            train_data_description,
        )


if __name__ == "__main__":
    main()

