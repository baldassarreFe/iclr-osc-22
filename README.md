# ICLR workshop

https://objects-structure-causality.github.io/

## Setup

Clone the repo:
```bash
git clone 'https://github.com/baldassarreFe/iclr-osc-22.git'
cd iclr-osc-22
```

Environment from scratch:
```bash
ENV_NAME='iclr-osc-22'
conda create -y -n "${ENV_NAME}" -c pytorch -c conda-forge \
    python=3.9 black isort pytest pre-commit \
    hydra-core colorlog tqdm wandb sphinx \
    'numpy>=1.20' pandas matplotlib seaborn tabulate scikit-learn scikit-image \
    jupyterlab=3 jupyterlab_code_formatter jupyter_console \
    tensorflow-cpu pytorch torchvision einops opt_einsum cudatoolkit-dev cudnn

conda activate "${ENV_NAME}"

python -m pip install \
    timm better_exceptions \
    sphinx-rtd-theme sphinx-autodoc-typehints \
    hydra_colorlog namesgenerator \
    'git+https://github.com/deepmind/multi_object_datasets'
conda env config vars set BETTER_EXCEPTIONS=1
pre-commit install

python -m pip install --editable .
```

Environment from file:
```bash
ENV_NAME='iclr-osc-22'
conda env create -n "${ENV_NAME}" -f 'environment.yaml'
conda activate "${ENV_NAME}"
pre-commit install
python -m pip install --editable .
```

# Datasets

Download ([other options](https://github.com/deepmind/multi_object_datasets)):
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
- [MultidSprites](notebooks/datasets/MultidSprites.ipynb)
- [ObjectsRoom](notebooks/datasets/ObjectsRoom.ipynb)
- [Tetrominoes](notebooks/datasets/Tetrominoes.ipynb)
- [ClevrWithMasks](notebooks/datasets/ClevrWithMasks.ipynb)
