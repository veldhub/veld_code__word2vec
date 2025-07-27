import pickle
import os
import sys

import gensim
import psycopg
from psycopg.sql import SQL, Identifier, Literal


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/"
IN_MODEL_FILE = os.getenv("in_model_file")
OUT_VECTOR_FILE = os.getenv("out_vector_file")
OUT_VECTOR_TABLE = os.getenv("out_vector_table")
OUT_PREFIX = os.getenv("out_prefix")
OUT_SUFFIX = os.getenv("out_suffix")
EXPORT_METHOD = sys.argv[1]
print("export method:", EXPORT_METHOD)
print("in_model_file:", IN_MODEL_FILE)
print("out_prefix:", OUT_PREFIX)
print("out_suffix:", OUT_SUFFIX)
if EXPORT_METHOD == "to_pkl":
    print("out_vector_file:", OUT_VECTOR_FILE)
elif EXPORT_METHOD == "to_sql":
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    print("out_vector_table:", OUT_VECTOR_TABLE)
    print("DB_NAME:", DB_NAME)
    print("DB_USER:", DB_USER)
    print("DB_PASSWORD:", DB_PASSWORD)
    print("DB_HOST:", DB_HOST)
    print("DB_PORT:", DB_PORT)
    conn = None
    cursor = None


def create_out_id_from_in_file(in_file):
    dot_pos = in_file.rfind(".")
    if dot_pos != -1:
        in_file_name = in_file[:dot_pos]
    else:
        in_file_name = in_file
    out_id = in_file_name
    if OUT_PREFIX:
        out_id = OUT_PREFIX + out_id
    if OUT_SUFFIX:
        out_id = out_id + OUT_SUFFIX
    if EXPORT_METHOD == "to_pkl":
        out_id = OUT_FOLDER + out_id + ".pkl"
    return out_id


def create_in_out_list():
    in_out_list = []
    if IN_MODEL_FILE:
        in_model_path = IN_FOLDER + IN_MODEL_FILE
        if OUT_VECTOR_FILE:
            out_vector_id = OUT_FOLDER + OUT_VECTOR_FILE
        elif OUT_VECTOR_TABLE:
            out_vector_id = OUT_VECTOR_TABLE
        else:
            out_vector_id = create_out_id_from_in_file(IN_MODEL_FILE)
        in_out_list.append((in_model_path, out_vector_id))
    else:
        for in_file in sorted(os.listdir(IN_FOLDER)):
            if in_file.endswith(".bin"):
                in_path = IN_FOLDER + in_file
                out_id = create_out_id_from_in_file(in_file)
                in_out_list.append((in_path, out_id))
    return in_out_list


def prepare_db():
    global conn
    global cursor
    conn = psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )
    conn.autocommit = True
    cursor = conn.cursor()
    sql_command = "SELECT version();"
    print(sql_command)
    cursor.execute(sql_command)
    print("connected to:", cursor.fetchone())
    sql_command = "CREATE EXTENSION IF NOT EXISTS vector;"
    print(sql_command)
    cursor.execute(sql_command)


def export_model(in_path, out_id):
    print(f"export_model: in_path: {in_path}, out_id: {out_id}")

    # loading model
    print("loading model")
    model = gensim.models.Word2Vec.load(in_path)
    
    # transforming vectors to dict
    print("transforming vectors to dict")
    vector_dict = {}
    for word, index in model.wv.key_to_index.items():
        vector_dict[word] = model.wv[index]
    print("len(vector_dict):", len(vector_dict))
    
    # pickle
    if EXPORT_METHOD == "to_pkl":
        print("persisting dict to pickle")
        with open(out_id, "wb") as f:
            pickle.dump(vector_dict, f)

    # sql
    elif EXPORT_METHOD == "to_sql":
        print("persisting dict to database")

        # perpare data
        word_embedding_list = []
        embedding_dim = None
        for word, embedding in vector_dict.items():
            if embedding_dim is None:
                embedding_dim = len(embedding)
            word_embedding_list.append((word, embedding.tolist()))

        # set up table
        query = SQL("DROP TABLE IF EXISTS {table_name}").format(table_name=Identifier(out_id))
        print(query.as_string())
        cursor.execute(query)
        query = SQL(
            "CREATE TABLE {table_name} ("
            "word TEXT PRIMARY KEY, "
            "embedding VECTOR({vector_dim}) not null"
            ")"
        ).format(
            table_name=Identifier(out_id),
            vector_dim=Literal(embedding_dim),
        )
        print(query.as_string())
        cursor.execute(query)

        # insert data
        query = SQL("INSERT INTO {table_name} (word, embedding) VALUES (%s, %s)").format(
            table_name=Identifier(out_id),
        )
        print(query.as_string())
        cursor.executemany(query, word_embedding_list)

        # verify insertion
        query = SQL("SELECT COUNT(*) FROM {table_name}").format(table_name=Identifier(out_id))
        print(query.as_string())
        cursor.execute(query)
        print(cursor.fetchone())



def main():
    if EXPORT_METHOD == "to_sql":
        prepare_db()
    for in_out in create_in_out_list():
        export_model(in_out[0], in_out[1])
    if EXPORT_METHOD == "to_sql":
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()

