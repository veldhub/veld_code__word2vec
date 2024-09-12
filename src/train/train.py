import os
import subprocess
from datetime import datetime

import gensim
import yaml
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec


# train data 
TRAIN_DATA_PATH = "/veld/input/" + os.getenv("in_train_data_file")

# model data
TRAINING_ARCHITECTURE = "word2vec"
MODEL_DESCRIPTION = os.getenv("model_description")
OUT_MODEL_FILE = os.getenv("out_model_file")
OUT_MODEL_PATH = "/veld/output/" + OUT_MODEL_FILE
OUT_MODEL_METADATA_PATH = "/veld/output/veld.yaml"
MODEL_ID = OUT_MODEL_FILE.replace(".bin", "")

# model hyperparameters
EPOCHS = int(os.getenv("epochs"))
VECTOR_SIZE = int(os.getenv("vector_size"))
WINDOW = int(os.getenv("window"))
MIN_COUNT = int(os.getenv("min_count"))

# dynamically loaded metadata
TRAIN_DATA_DESCRIPTION = None
DURATION = None


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
            global TRAIN_DATA_DESCRIPTION
            try:
                TRAIN_DATA_DESCRIPTION = input_veld_metadata["x-veld"]["data"]["description"]
            except:
                pass


def print_params():
    print(f"TRAIN_DATA_PATH: {TRAIN_DATA_PATH}", flush=True)
    print(f"TRAIN_DATA_DESCRIPTION: {TRAIN_DATA_DESCRIPTION}", flush=True)
    print(f"OUT_MODEL_PATH: {OUT_MODEL_PATH}", flush=True)
    print(f"OUT_MODEL_PATH: {OUT_MODEL_PATH}", flush=True)
    print(f"EPOCHS: {EPOCHS}", flush=True)
    print(f"TRAINING_ARCHITECTURE: {TRAINING_ARCHITECTURE}", flush=True)
    print(f"VECTOR_SIZE: {VECTOR_SIZE}", flush=True)
    print(f"WINDOW: {WINDOW}", flush=True)
    print(f"MIN_COUNT: {MIN_COUNT}", flush=True)


def train_and_persist():

    class LossLogger(CallbackAny2Vec):
        def __init__(self):
            self.epoch = 0

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            print(f"epoch: {self.epoch}, loss: {loss}", flush=True)
            self.epoch += 1

    print("start training", flush=True)
    sentences = LineSentence(TRAIN_DATA_PATH)
    time_start = datetime.now()
    model = gensim.models.Word2Vec(
        sentences=sentences,
        epochs=EPOCHS,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=os.cpu_count(),
        callbacks=[LossLogger()],
    )
    global DURATION
    DURATION = (datetime.now() - time_start).seconds / 60
    model.save(OUT_MODEL_PATH)
    print(f"done. duration in minutes: {DURATION}", flush=True)


def write_metadata():

    # calculate size of training and model data
    def calc_size(file_or_folder):
        size = subprocess.run(["du", "-sh", file_or_folder], capture_output=True, text=True)
        size = size.stdout.split()[0]
        return size
    train_data_size = calc_size(TRAIN_DATA_PATH)
    model_data_size = calc_size(OUT_MODEL_PATH)

    # calculate hash of training data
    train_data_md5_hash = subprocess.run(["md5sum", TRAIN_DATA_PATH], capture_output=True, text=True)
    train_data_md5_hash = train_data_md5_hash.stdout.split()[0]

    # aggregate into metadata dictionary
    out_veld_metadata = {
        "x-veld": {
            "data": {
                "description": MODEL_DESCRIPTION,
                "file_types": "bin",
                "topics": [
                    "NLP",
                    "word embeddings",
                ],
                "contents": [
                    "word embeddings model",
                    "word2vec model",
                ],
                "additional": {
                    "model_id": MODEL_ID,
                    "training_architecture": TRAINING_ARCHITECTURE,
                    "train_data_description": TRAIN_DATA_DESCRIPTION,
                    "train_data_size": train_data_size,
                    "train_data_md5_hash": train_data_md5_hash,
                    "training_vector_size": VECTOR_SIZE,
                    "window": WINDOW,
                    "min_count": MIN_COUNT,
                    "training_duration (minutes)": round(DURATION, 1),
                    "model_data_size": model_data_size,
                }
            }
        }
    }

    # write to yaml
    with open(OUT_MODEL_METADATA_PATH, "w") as f:
        yaml.dump(out_veld_metadata, f, sort_keys=False)


def main():
    get_description()
    print_params()
    train_and_persist()
    write_metadata()


if __name__ == "__main__":
    main()

