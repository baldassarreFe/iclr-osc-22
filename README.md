# Towards self-supervised learning of global and object-centric representations

For the ICLR workshop on [Objects, Structure, and Causality](https://objects-structure-causality.github.io/).

## Setup

First, clone the repo:
```bash
git clone 'https://github.com/baldassarreFe/iclr-osc-22.git'
cd iclr-osc-22
```

Create an environment from scratch:
```bash
ENV_NAME='iclr-osc-22'
conda create -y -n "${ENV_NAME}" -c pytorch -c conda-forge \
    python=3.9 black isort pytest pre-commit \
    hydra-core colorlog submitit tqdm wandb sphinx \
    'numpy>=1.20' pandas matplotlib seaborn tabulate scikit-learn scikit-image \
    jupyterlab=3 jupyterlab_code_formatter jupyter_console \
    tensorflow-cpu pytorch torchvision einops opt_einsum cudatoolkit-dev cudnn

conda activate "${ENV_NAME}"

python -m pip install \
    timm better_exceptions \
    sphinx-rtd-theme sphinx-autodoc-typehints \
    hydra_colorlog hydra-submitit-launcher namesgenerator \
    'git+https://github.com/deepmind/multi_object_datasets'
conda env config vars set BETTER_EXCEPTIONS=1
pre-commit install

python -m pip install --editable .
```

Or create an environment using the provided dependency file:
```bash
ENV_NAME='iclr-osc-22'
conda env create -n "${ENV_NAME}" -f 'environment.yaml'
conda activate "${ENV_NAME}"
pre-commit install
python -m pip install --editable .
```

## Datasets

The project uses the "CLEVR with masks dataset", which is part of the
[Multi Object Datasets](https://github.com/deepmind/multi_object_datasets) collection.

Download all datasets from a Google Cloud bucket (see original website for other options):
```bash
sudo apt install -y apt-transport-https ca-certificates gnupg
echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' |
    sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl 'https://packages.cloud.google.com/apt/doc/apt-key.gpg' |
    sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt update
sudo apt install -y google-cloud-sdk

gsutil -m cp -r gs://multi-object-datasets "${HOME}/"
```

Data loading and visualization notebooks:
- [Multi-dSprites](notebooks/datasets/MultidSprites.ipynb)
- [ObjectsRoom](notebooks/datasets/ObjectsRoom.ipynb)
- [Tetrominoes](notebooks/datasets/Tetrominoes.ipynb)
- [CLEVR with masks](notebooks/datasets/ClevrWithMasks.ipynb)

Prepare the CLEVR dataset for training and evaluation by splitting the original
TFRecords file in 3 parts (train+val only contain RGB images, test contains
the full sample dict with object masks and attributes):
```bash
python -m osc.data.clevr_with_masks --data-root "${HOME}/multi-object-datasets"
```

## Training

A training run with default parameters can be launched by executing following commands
from the root of the repository (local training, single GPU):
```bash
export CUDA_VISIBLE_DEVICES=0
./train.py
```

The project uses [Hydra](https://hydra.cc/) as a configuration manager. All defaults
can be listed as:
```bash
./train.py --cfg job
```

Individual parameters from the configuration can be changed on the command line:
```bash
./train.py \
  training.batch_size=16 \
  model.backbone.embed_dim=64 \
  model.backbone.patch_size='[4,4]' \
  model.backbone.embed_dim=64 \
  model.backbone.num_heads=8 \
  model.backbone.{block_drop,block_attn_drop}=0.2
```

All available configuration groups (e.g. different loss functions, attention types,
learning rate schedules, etc.) can be found in the [`configs` folder](./configs).
For example, to train with an object-wise contrastive loss that takes all object
tokens from all images as negatives and overfit on a small subset of images:
```bash
./train.py losses/l_objects=ctr_all +overfit=overfit
```

Running a parameter sweep in a SLURM environment is also supported, for example:
```bash
./train.py --multirun hydra/launcher=submitit_slurm +slurm=slurm \
  +losses=more_objects,more_global \
  model=vit_obj_global \
  model/obj_queries=sample \
  model.backbone.embed_dim=64,128,256 \
  logging.group='slurm_sweep' \
  lr_scheduler=linear1_cosine4_x5 \
  lr_scheduler.decay.end_lr=0.0003 \
  optimizer.start_lr=0.0007 \
  optimizer.weight_decay=0.0001 \
  model.backbone.num_heads=4,8 \
  model.backbone.num_layers=2,4,6 \
  model.obj_fn.num_iters=1,2,4
```

## Hyperparameters

Here follow the main hyperparameters that can be configured for the experiments.
A corresponding configuration file can be found  in the [`configs` folder](./configs).

Architectures:
- `backbone-global_fn-global_proj`:
  global representation only.
  Backbone patch tokens can be aggregated either with global average pooling (`avg`)
  or an extra CLS token (`cls`)
- `backbone(-slot_fn-slot_proj)-global_fn-global_proj`:
  after the backbone, two separate branches process global and object features.
  Backbone patch tokens can be aggregated either with global average pooling (`avg`)
  or an extra CLS token (`cls`)
- `backbone-slot_fn(-global_fn-global_proj)-slot_proj`:
  after the backbone, the slot function extracts `S` object representations,
  these `S` feature tokens are further projected to yield object representations,
  furthermore these `S` tokens are average-pooled and processed to extract global
  features and projections.
  The backbone pooling is set to `avg` since a CLS token would not be ignored.

Object query implementations:
- `learned`: learned query tokens in fixed number
- `sample`: object queries are sampled either from a single Gaussian distributions
  with learned parameters, or a mixture of Gaussiand with uniform component weights
- `kmeans_euclidean`: object queries are initialized as the K-Means clustering of
  backbone features. Number of clusters can be dynamically chosen, the distance function
  is a simply Euclidean distance.

Object function implementations:
- `slot-attention` slot attention decoder (iterative)
- `cross-attention` cross attention decoder
- co attention decoder (not implemented yet)

Loss functions:
- Global image representation:
  - Contrastive loss `ctr`
    (given one image, classify positively an augmented version of that image
    among `B-2` other unrelated images in the batch)
  - Cosine similarity loss `sim`
    (given one image and its augmented version, maximise the cosine similarity
    between their projected representations)
- Object representation:
  - Contrastive loss `ctr_all`
    (one token compared to all tokens in all images)
  - Contrastive loss `ctr_img`
    (one token compared to all tokens from its original image and the augmented version)
  - Cosine similarity loss `sim_img`
    (one token compared to all tokens from its original image and the augmented version)

Embedding dimension:
- default 64 for everything with 2x factor for all MLP hidden layers
- 128 and 256 also work well but require a smaller batch size especially
  when using 8 heads. Safe (dim, batch) pairs: (64, 64), (128, 16), (256, 8)
- Interesting to try different size for the final projection head when using
  matching cosine similarity loss

## Documentation

The documentation is hosted on a
[GitHub Pages website](https://baldassarrefe.github.io/iclr-osc-22/).

The documentation is generated automatically using Sphinx. All sources are in the
[`docs/` folder](./docs). A separate `docs` branch tracks the documentation builds.
