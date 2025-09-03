import csv
import pickle
import os
import sys
from textwrap import dedent

import gensim
import psycopg
import yaml
from psycopg.sql import SQL, Identifier, Literal


IN_FOLDER = "/veld/input/"
OUT_FOLDER = "/veld/output/"
IN_MODEL_FILE = os.getenv("in_model_file")
IN_METADATA_FILE = os.getenv("in_metadata_file")
OUT_METADATA_FILE = os.getenv("out_metadata_file")
ENABLE_INCLUSION_METADATA = os.getenv("enable_inclusion_metadata")
if ENABLE_INCLUSION_METADATA == "true":
    ENABLE_INCLUSION_METADATA = True
else:
    ENABLE_INCLUSION_METADATA = False
OUT_PKL_FILE = os.getenv("out_pkl_file")
OUT_CSV_FILE = os.getenv("out_csv_file")
OUT_VECTOR_TABLE = os.getenv("out_vector_table")
OUT_METADATA_TABLE = os.getenv("out_metadata_table")
OUT_PREFIX = os.getenv("out_prefix")
OUT_SUFFIX = os.getenv("out_suffix")
EXPORT_METHOD = sys.argv[1]
print("export method:", EXPORT_METHOD)
print("in_model_file:", IN_MODEL_FILE)
print("in_metadata_file:", IN_METADATA_FILE)
print("enable_inclusion_metadata:", ENABLE_INCLUSION_METADATA)
print("out_prefix:", OUT_PREFIX)
print("out_suffix:", OUT_SUFFIX)
if EXPORT_METHOD == "to_pkl":
    print("out_pkl_file:", OUT_PKL_FILE)
    print("out_metadata_file:", OUT_METADATA_FILE)
elif EXPORT_METHOD == "to_csv":
    print("out_csv_file:", OUT_CSV_FILE)
    print("out_metadata_file:", OUT_METADATA_FILE)
elif EXPORT_METHOD == "to_db":
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    print("out_vector_table:", OUT_VECTOR_TABLE)
    print("out_metadata_table:", OUT_METADATA_TABLE)
    print("DB_NAME:", DB_NAME)
    print("DB_USER:", DB_USER)
    print("DB_PASSWORD:", DB_PASSWORD)
    print("DB_HOST:", DB_HOST)
    print("DB_PORT:", DB_PORT)
    conn = None
    cursor = None
    query = None


class Query:
    def __init__(self, cur):
        self.cur = cur

    def __call__(self, query, print_query=False, **kwargs):
        kwargs_cleaned = {}
        for key, value in kwargs.items():
            if type(value) is type(self):
                kwargs_cleaned[key] = value.as_sql()
            else:
                kwargs_cleaned[key] = value
        self.query = SQL(query).format(**kwargs_cleaned)
        if print_query:
            print(self.query.as_string())
        return self

    def as_sql(self):
        return self.query

    def execute(self, data=None):
        if data:
            self.cur.execute(self.query, data)
        else:
            self.cur.execute(self.query)
        self.query = None
        return self

    def executemany(self, data):
        self.cur.executemany(self.query, data)
        return self

    def fetchall(self):
        return self.cur.fetchall()


def prepare_db():
    global conn
    global cursor
    global query
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
    query = Query(cur=cursor)


def create_in_list():
    in_list = []
    if IN_MODEL_FILE:
        in_list.append(
            (
                IN_MODEL_FILE,
                IN_METADATA_FILE,
                IN_FOLDER + IN_MODEL_FILE,
                IN_FOLDER + IN_METADATA_FILE,
            )
        )
    else:
        for in_model_file in sorted(os.listdir(IN_FOLDER)):
            if in_model_file.endswith(".bin"):
                in_metadata_file = in_model_file.replace(".bin", ".yaml")
                if not os.path.exists(IN_FOLDER + in_metadata_file):
                    in_metadata_file = None
                in_list.append(
                    (
                        in_model_file,
                        in_metadata_file,
                        IN_FOLDER + in_model_file,
                        IN_FOLDER + in_metadata_file,
                    )
                )
    return in_list


def create_out_target(in_model_file):
    dot_pos = in_model_file.rfind(".")
    if dot_pos != -1:
        in_file_name = in_model_file[:dot_pos]
    else:
        in_file_name = in_model_file
    out_target = in_file_name
    if OUT_PREFIX:
        out_target = OUT_PREFIX + out_target
    if OUT_SUFFIX:
        out_target = out_target + OUT_SUFFIX
    return out_target


def load_model_and_metadata(in_model_path, in_metadata_path):
    print(
        f"load_model_and_metadata: in_model_path: {in_model_path}, in_metadata_path:"
        f" {in_metadata_path}"
    )

    # loading model
    print("loading model")
    model = gensim.models.Word2Vec.load(in_model_path)

    # transforming vectors to dict
    print("transforming vectors to dict")
    vector_dict = {}
    for word, index in model.wv.key_to_index.items():
        vector_dict[word] = model.wv[index]
    print("len(vector_dict):", len(vector_dict))

    # loading metadata
    if ENABLE_INCLUSION_METADATA and in_metadata_path:
        with open(in_metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = None

    return vector_dict, metadata


def export_model_to_pkl(vector_dict, metadata, out_pkl_path, out_metadata_path):
    print(
        f"export_model_to_pkl: out_pkl_path: {out_pkl_path}, out_metadata_path: {out_metadata_path}"
    )
    with open(out_pkl_path, "wb") as f:
        pickle.dump(vector_dict, f)
    if metadata:
        metadata["x-veld"]["data"]["file_type"] = "pkl"
        with open(out_metadata_path, "w") as f:
            yaml.dump(metadata, f, sort_keys=False)


def export_model_to_csv(vector_dict, metadata, out_csv_path, out_metadata_path):
    print(
        f"export_model_to_csv: out_csv_path: {out_csv_path}, out_metadata_path: {out_metadata_path}"
    )
    with open(out_csv_path, "w") as f:
        writer = csv.writer(f)
        header_is_written = False
        for word, embedding in vector_dict.items():
            if not header_is_written:
                dim_list = [f"dim_{i}" for i in range(1, len(embedding) + 1)]
                writer.writerow(["word"] + dim_list)
                header_is_written = True
            writer.writerow([word] + embedding.tolist())
    if metadata:
        metadata["x-veld"]["data"]["file_type"] = "csv"
        with open(out_metadata_path, "w") as f:
            yaml.dump(metadata, f, sort_keys=False)


def convert_data_to_insert_rows(vector_dict, metadata):
    word_embedding_list = []
    embedding_dim = None
    for word, embedding in vector_dict.items():
        if embedding_dim is None:
            embedding_dim = len(embedding)
        if metadata:
            word_embedding_list.append(
                (
                    word,
                    embedding.tolist(),
                    metadata["model_id"],
                    metadata["training_architecture"],
                )
            )
        else:
            word_embedding_list.append((word, embedding.tolist()))
    return word_embedding_list, embedding_dim


def write_metadata_table(metadata_table_name, metadata, vector_table_name):
    print("write_metadata_table")

    # define table
    query(
        dedent(
            """\
            CREATE TABLE IF NOT EXISTS {table_name} (
              vector_table_name TEXT NOT NULL,
              model_id TEXT NOT NULL,
              training_architecture TEXT NOT NULL,
              train_data_description TEXT,
              train_data_size TEXT,
              train_data_md5_hash TEXT,
              training_vector_size INTEGER,
              training_epochs INTEGER,
              window_size INTEGER,
              min_count INTEGER,
              training_duration_minutes REAL,
              model_data_size TEXT,
              PRIMARY KEY (model_id, training_architecture)
            )
            """
        ),
        print_query=True,
        table_name=Identifier(metadata_table_name),
    ).execute()

    # insert metadata
    query(
        dedent(
            """\
            INSERT INTO {table_name}
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id, training_architecture) DO UPDATE SET
              vector_table_name = EXCLUDED.vector_table_name,
              train_data_description = EXCLUDED.train_data_description,
              train_data_size = EXCLUDED.train_data_size,
              train_data_md5_hash = EXCLUDED.train_data_md5_hash,
              training_vector_size = EXCLUDED.training_vector_size,
              training_epochs = EXCLUDED.training_epochs,
              window_size = EXCLUDED.training_epochs,
              min_count = EXCLUDED.training_epochs,
              training_duration_minutes = EXCLUDED.training_duration_minutes,
              model_data_size = EXCLUDED.training_duration_minutes
            """
        ),
        print_query=True,
        table_name=Identifier(metadata_table_name),
    ).execute(
        (
            vector_table_name,
            metadata["model_id"],
            metadata["training_architecture"],
            metadata["train_data_description"],
            metadata["train_data_size"],
            metadata["train_data_md5_hash"],
            metadata["training_vector_size"],
            metadata["training_epochs"],
            metadata["window"],
            metadata["min_count"],
            metadata["training_duration (minutes)"],
            metadata["model_data_size"],
        )
    )


def write_vector_table(out_vector_table, metadata_table_name, vector_dict, metadata):
    print("write_vector_table")

    # convert data
    word_embedding_list, embedding_dim = convert_data_to_insert_rows(vector_dict, metadata)

    # set up table
    query(
        "DROP TABLE IF EXISTS {table_name}",
        print_query=True,
        table_name=Identifier(out_vector_table),
    ).execute()

    # define table
    if metadata:
        query(
            dedent(
                """\
                CREATE TABLE {table_name} (
                  word TEXT PRIMARY KEY,
                  embedding VECTOR({vector_dim}) NOT NULL,
                  model_id TEXT NOT NULL,
                  training_architecture TEXT NOT NULL,
                  FOREIGN KEY (model_id, training_architecture) REFERENCES {models_metadata_table_name} (model_id, training_architecture)
                )
                """
            ),
            print_query=True,
            table_name=Identifier(out_vector_table),
            models_metadata_table_name=Identifier(metadata_table_name),
            vector_dim=Literal(embedding_dim),
        ).execute()
    else:
        query(
            dedent(
                """\
                CREATE TABLE {table_name} (
                    word TEXT PRIMARY KEY,
                    embedding VECTOR({vector_dim}) NOT NULL
                )
                """
            ),
            print_query=True,
            table_name=Identifier(out_vector_table),
            vector_dim=Literal(embedding_dim),
        ).execute()

    # insert
    if metadata:
        query(
            dedent(
                """\
                INSERT INTO {table_name}
                (word, embedding, model_id, training_architecture)
                VALUES (%s, %s, %s, %s)
                """
            ),
            print_query=True,
            table_name=Identifier(out_vector_table),
        ).executemany(word_embedding_list)
    else:
        query(
            dedent(
                """\
                INSERT INTO {table_name}
                (word, embedding)
                VALUES (%s, %s)
                """
            ),
            print_query=True,
            table_name=Identifier(out_vector_table),
        ).executemany(word_embedding_list)


def export_model_to_db(vector_dict, metadata, out_vector_table, out_metadata_table):
    print(
        f"export_model_to_db: out_vector_table: {out_vector_table}, out_metadata_table:"
        f" {out_metadata_table}"
    )

    # set up metadata
    if metadata:
        metadata = metadata["x-veld"]["data"]["additional"]
        write_metadata_table(
            metadata_table_name=out_metadata_table,
            metadata=metadata,
            vector_table_name=out_vector_table,
        )
    else:
        metadata = None

    # write data
    write_vector_table(
        out_vector_table,
        out_metadata_table,
        vector_dict,
        metadata,
    )

    # verify insertion
    query("SELECT COUNT(*) FROM {table_name}", table_name=Identifier(out_vector_table)).execute()
    print(query.fetchall())


def main():
    if EXPORT_METHOD == "to_db":
        prepare_db()
    for in_model_file, in_metadata_file, in_model_path, in_metadata_path in create_in_list():
        vector_dict, metadata = load_model_and_metadata(in_model_path, in_metadata_path)
        if EXPORT_METHOD == "to_pkl" or EXPORT_METHOD == "to_csv":
            if ENABLE_INCLUSION_METADATA and OUT_METADATA_FILE:
                out_metadata_path = OUT_FOLDER + OUT_METADATA_FILE
            elif ENABLE_INCLUSION_METADATA and in_metadata_file:
                out_metadata_path = OUT_FOLDER + in_metadata_file
            else:
                out_metadata_path = None
            if EXPORT_METHOD == "to_pkl":
                if OUT_PKL_FILE:
                    out_pkl_path = OUT_FOLDER + OUT_PKL_FILE
                else:
                    out_pkl_path = OUT_FOLDER + create_out_target(in_model_file) + ".pkl"
                export_model_to_pkl(
                    vector_dict=vector_dict,
                    metadata=metadata,
                    out_pkl_path=out_pkl_path,
                    out_metadata_path=out_metadata_path,
                )
            elif EXPORT_METHOD == "to_csv":
                if OUT_CSV_FILE:
                    out_csv_path = OUT_FOLDER + OUT_CSV_FILE
                else:
                    out_csv_path = OUT_FOLDER + create_out_target(in_model_file) + ".csv"
                export_model_to_csv(
                    vector_dict=vector_dict,
                    metadata=metadata,
                    out_csv_path=out_csv_path,
                    out_metadata_path=out_metadata_path,
                )
        elif EXPORT_METHOD == "to_db":
            if ENABLE_INCLUSION_METADATA and OUT_METADATA_TABLE:
                out_metadata_table = OUT_METADATA_TABLE
            elif ENABLE_INCLUSION_METADATA and in_metadata_file:
                out_metadata_table = "models_metadata"
            else:
                out_metadata_table = None
            if OUT_VECTOR_TABLE:
                out_vector_table = OUT_VECTOR_TABLE
            else:
                out_vector_table = create_out_target(in_model_file)
            export_model_to_db(
                vector_dict=vector_dict,
                metadata=metadata,
                out_vector_table=out_vector_table,
                out_metadata_table=out_metadata_table,
            )
    if EXPORT_METHOD == "to_db":
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
