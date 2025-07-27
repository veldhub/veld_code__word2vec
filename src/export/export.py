import pickle
import os

import gensim


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/"
IN_MODEL_FILE = os.getenv("in_model_file")
OUT_VECTOR_FILE = os.getenv("out_vector_file")
print("IN_MODEL_FILE:", IN_MODEL_FILE)
print("OUT_VECTOR_FILE:", OUT_VECTOR_FILE)


def create_out_path_from_in_file(in_file):
    dot_pos = in_file.rfind(".")
    if dot_pos != -1:
        in_file = in_file[:dot_pos]
    else:
        in_file = in_file
    out_path = OUT_FOLDER + in_file + ".pkl"
    return out_path


def create_in_out_path_list():
    in_out_path_list = []
    if IN_MODEL_FILE:
        in_model_path = IN_FOLDER + IN_MODEL_FILE
        if OUT_VECTOR_FILE:
            out_vector_path = OUT_FOLDER + OUT_VECTOR_FILE
        else:
            out_vector_path = create_out_path_from_in_file(IN_MODEL_FILE)
        in_out_path_list.append((in_model_path, out_vector_path))
    else:
        for in_file in sorted(os.listdir(IN_FOLDER)):
            if in_file.endswith(".bin"):
                in_path = IN_FOLDER + in_file
                out_path = create_out_path_from_in_file(in_file)
                in_out_path_list.append((in_path, out_path))
    return in_out_path_list


def export_model(in_path, out_path):
    print(f"export_model: in_path: {in_path}, out_path: {out_path}")

    # loading model
    print("loading model")
    model = gensim.models.Word2Vec.load(in_path)

    # transforming vectors to dict
    print("transforming vectors to dict")
    vector_dict = {}
    for word, index in model.wv.key_to_index.items():
        vector_dict[word] = model.wv[index]

    # persisting dict to pickle
    print("persisting dict to pickle")
    with open(out_path, "wb") as f:
        pickle.dump(vector_dict, f)


def main():
    for in_out_path in create_in_out_path_list():
        export_model(in_out_path[0], in_out_path[1])


if __name__ == "__main__":
    main()

