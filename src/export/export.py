import pickle
import os

import gensim


IN_MODEL_FILE = "/veld/input/" + os.getenv("in_model_file")
OUT_VECTOR_FILE = "/veld/output/" + os.getenv("out_vector_file")
print("IN_MODEL_FILE:", IN_MODEL_FILE)
print("OUT_VECTOR_FILE:", OUT_VECTOR_FILE)


# loading model
print("loading model")
model = gensim.models.Word2Vec.load(IN_MODEL_FILE)

# transforming vectors to dict
print("transforming vectors to dict")
vector_dict = {}
for word, index in model.wv.key_to_index.items():
    vector_dict[word] = model.wv[index]

# persisting dict to pickle
print("persisting dict to pickle")
with open(OUT_VECTOR_FILE, "wb") as f:
    pickle.dump(vector_dict, f)

