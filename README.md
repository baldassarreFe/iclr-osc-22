# ICLR workshop

https://objects-structure-causality.github.io/

## Setup

From scratch:
```bash
ENV_NAME='iclr-osc-22'
conda create -y -n "${ENV_NAME}" -c pytorch -c conda-forge \
    python=3.9 black isort pytest pre-commit \
    numpy pandas matplotlib seaborn scikit-learn \
    jupyterlab=3 jupyterlab_code_formatter jupyter_console \
    tabulate tqdm \
    pytorch torchvision einops cudatoolkit-dev cudnn

conda activate "${ENV_NAME}"

python -m pip install timm slot_attention better_exceptions
conda env config vars set BETTER_EXCEPTIONS=1
pre-commit install

python -m pip install --editable .
```

From file:
```bash
ENV_NAME='iclr-osc-22'
conda env create -n "${ENV_NAME}" -f 'environment.yaml'
conda activate "${ENV_NAME}"
pre-commit install
python -m pip install --editable .
```
