import logging
import subprocess
from pathlib import Path
from typing import Any, MutableMapping, Union

import wandb
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

EXCLUDE_HPARAMS = (
    "data.root",
    "data.normalize",
    "data.train.seed",
    "data.val.seed",
    "other",
    "logging",
)


def setup_wandb(cfg):
    """Create wandb run, log hyperparams, auto upload checkpoints"""
    hparams = filter_cfg_for_wandb(cfg)
    wandb.init(
        project=cfg.logging.project,
        group=cfg.logging.group,
        id=cfg.logging.id,
        name=cfg.logging.name,
        tags=cfg.logging.tags,
        notes=cfg.logging.notes,
        config=hparams,
        mode="online" if not cfg.other.debug else "disabled",
    )
    wandb.save("checkpoint.*.pth", policy="live")
    wandb.save("*.yaml", policy="live")
    if cfg.other.debug:
        log.info("Debug run, wandb disabled")


def filter_cfg_for_wandb(cfg, exclude=None):
    def delete_(d: MutableMapping[str, Any], k: str):
        k = k.split(".", maxsplit=1)
        if len(k) == 1:
            del d[k[0]]
        else:
            delete_(d[k[0]], k[1])

    cfg = OmegaConf.to_container(cfg, resolve=True)
    if exclude is None:
        exclude = EXCLUDE_HPARAMS
    for k in exclude:
        delete_(cfg, k)
    return cfg


def find_run_by_name(name, output_dir: Union[Path, str] = None) -> Path:
    p = subprocess.run(
        ["grep", name, "--include=train.yaml", "-r", ".", "--files-with-matches"],
        capture_output=True,
        text=True,
        cwd=output_dir,
    )
    outputs = p.stdout.splitlines()
    if len(outputs) == 0:
        raise FileNotFoundError(name)
    if len(outputs) > 1:
        raise RuntimeError(f"Multiple matches {name}: {outputs}")
    return Path(outputs[0]).parent
