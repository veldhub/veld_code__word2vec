import gensim


model_path = "/veld/data/veld_data_11_word2vec_models/data/test/test.bin"

model = gensim.models.Word2Vec.load(model_path)
    
print(model.wv["und"])

