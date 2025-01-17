# veld_code__word2vec

This repo contains [code velds](https://zenodo.org/records/13322913) encapsulating training and 
usage of word2vec models.

## requirements

- git
- docker compose (note: older docker compose versions require running `docker-compose` instead of 
  `docker compose`)

## how to use

A code veld may be integrated into a chain veld, or used directly by adapting the configuration 
within its yaml file and using the template folders provided in this repo. Open the respective veld 
yaml file for more information.

Run a veld with:
```
docker compose -f <VELD_NAME>.yaml up
```

## contained code velds

**[./veld_train.yaml](./veld_train.yaml)** 

Trains a model from scratch.

```
docker compose -f veld_train.yaml up
```

**[./veld_jupyter_notebook.yaml](./veld_jupyter_notebook.yaml)**

Launches an interactive jupyter notebook for playing with the models.

```
docker compose -f veld_jupyter_notebook.yaml up
```

