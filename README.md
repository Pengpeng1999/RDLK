# RDLK: Low-Rank Topology-Attribute Decoupling for Link Prediction

This repository contains the reference implementation of **RDLK**, a subgraph-based link prediction method that constructs topology and attribute features through low-rank decomposition and similar-subgraph retrieval.

The code is associated with the manuscript:

> Low-Rank Topology-Attribute Decoupling for Link Prediction

## Overview

RDLK follows a subgraph-based link prediction pipeline:

1. Split observed links into training and testing links.
2. Extract enclosing subgraphs around target node pairs.
3. Recover denoised/completed topology and attribute representations using low-rank decomposition.
4. Retrieve similar subgraphs to introduce long-distance topology/attribute information.
5. Concatenate constructed features and train a simple downstream classifier for link prediction.

## Repository Structure

```text
.
├── main.py              # Main entry point for running RDLK experiments
├── utils.py             # Data split, subgraph extraction, DRNL labeling, similarity search, feature construction
├── TRPCA_torch.py       # Torch-based low-rank / tensor RPCA utilities
├── rpca_ADMM.py         # ADMM-based RPCA solver
├── GNN.py               # Simple GCN/GAT modules used in auxiliary experiments
├── data/                # Preprocessed graph datasets in .pkl format
└── n2v/                 # Precomputed node2vec embeddings in .npy format
```

Each dataset file in `data/` is expected to be a pickle file containing:

```python
{
    "topo": scipy_sparse_adjacency_matrix,
    "attr": scipy_sparse_attribute_matrix
}
```

## Requirements

The implementation was developed with Python and common scientific computing libraries. A typical environment should include:

```bash
pip install numpy scipy scikit-learn pandas networkx tqdm psutil node2vec
pip install torch torch-geometric
```

Install the PyTorch and PyTorch Geometric versions that match your CUDA environment. See the official installation pages for platform-specific commands.

## Quick Start

Run an experiment from the repository root:

```bash
python main.py --data-name 2.cornell
```

Example with another dataset:

```bash
python main.py --data-name 9.cora --hop 2 --p 0.5 --gamma 1
```

The script prints AUC, accuracy, F1, precision, recall, and memory usage.

## Main Arguments

```text
--data-name            Dataset name without .pkl, e.g., 2.cornell or 9.cora
--test-ratio           Ratio of positive links used for testing
--max-train-num        Maximum number of positive training links
--hop                  Enclosing subgraph hop number
--p                    Threshold for binarizing the recovered topology matrix
--gamma                Balance parameter for low-rank decomposition
--max-nodes-per-hop    Maximum sampled nodes per hop
--use-embedding        Whether to concatenate node2vec embeddings
--use-similar          Whether to concatenate heuristic similarity features
--use-attribute        Whether to use node attributes
--init-attri           Whether to use initial pairwise attribute features
--no-cuda              Disable CUDA
```

## Available Datasets

The `data/` directory currently includes the following preprocessed datasets:

```text
1.bat
2.cornell
3.texas
4.washington
5.wisconsin
6.eat
7.uat
8.chameleon
9.cora
10.acm
11.uai2010
12.citeseer
13.dblp
14.wiki-cs
15.blogcatalog
16.flickr
17.amap
18.amac
19.corafull
20.pubmed
21.scholat
```

Use the dataset name without the `.pkl` suffix:

```bash
python main.py --data-name 12.citeseer
```

## Node2Vec Embeddings

Precomputed node2vec embeddings are stored in `n2v/`. By default, `--use-embedding` is disabled.

If `--use-embedding` is enabled, make sure the embedding path in `main.py` points to the local `n2v/` directory, or regenerate embeddings with `generate_node2vec_embeddings` in `utils.py`.

## Notes for Reproducibility

- Run commands from the repository root so that `data/` can be resolved correctly.
- The default train/test split uses an 80/20 positive-link split.
- Negative links are sampled from unobserved node pairs.
- The script currently caps the number of training links with `--max-train-num`.
- Some feature settings are controlled by commented lines in `main.py`; please keep the selected feature combination consistent when reproducing reported results.
