# Introduction
This Repo is based on [SimCSE](https://github.com/princeton-nlp/SimCSE), but specifically on its unsupervised learning framework for Contrastive Learning.

To reveal the magic behind BERT/RoBERTa as the extractor of embeddings, we replace the embedding extractor from BERT/roBERTa to albert, gemma and llama2. It's basically a comparison about bi-directional and uni-directional attension mechanism.


# Install & Train

For development, We used [poetry](https://python-poetry.org/), which is the dependency management and packaging tool for Python.

If you use poetry, you can install necessary packages by following command.

```bash
poetry install
```

Or, you can install them using `requiments.txt`.

```bash
pip install -r requirements.txt
```

The `requirements.txt` is output by following command.

```bash
poetry export -f requirements.txt --output requirements.txt
```

Then, you must execute `download.sh` to download training and evaluation datasets beforehand.
`download.sh` will collect STS and training datasets used in the paper in parallel.

```bash
bash download.sh
```

Finaly, you can train your model as below.

```bash
poetry run python train.py

# or
# python train.py
```


# Evaluation (Unsup-SimCSE)

In doing this implementation, we investigated how well the Unsup-SimCSE model trained by this implementation would perform.

We performed fine-tuning of Unsup-SimCSE **50 times** with different random seeds ([0, 49]) with the same dataset and hyperparameters as described in the paper (see `train-original.sh` and `train-multiple.sh`).