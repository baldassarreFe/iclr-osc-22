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
    """Create wandb run, save config, log, and hyperparameters"""
    if cfg.other.debug:
        mode = "disabled"
        log.info("Debug run, wandb disabled")
    else:
        mode = "online"
    wandb.init(
        project=cfg.logging.project,
        group=cfg.logging.group,
        id=cfg.logging.id,
        name=cfg.logging.name,
        tags=cfg.logging.tags,
        notes=cfg.logging.notes,
        config=filter_cfg_for_wandb(cfg),
        mode=mode,
    )
    wandb.save("train.yaml", policy="now")
    wandb.save("train.log", policy="live")
    wandb.save("checkpoint.*.pth", policy="live")


def filter_cfg_for_wandb(cfg, exclude=None):
    """Remove unwanted entries for wandb config."""

    def delete_(d: MutableMapping[str, Any], k: str):
        k = k.split(".", maxsplit=1)
        if len(k) == 1:
            del d[k[0]]
        else:
            delete_(d[k[0]], k[1])

    cfg = OmegaConf.to_container(cfg, resolve=True)
    if exclude is None:
        exclude = EXCLUDE_HPARAMS
    for key in exclude:
        delete_(cfg, key)
    return cfg


def find_run_by_name(name, output_dir: Union[Path, str] = None) -> Path:
    """Find a run by wandb name/id by grepping all train.yaml files under a folder"""
    p = subprocess.run(
        [
            "grep",
            "--include=train.yaml",
            "--exclude-dir=wandb",
            "--files-with-matches",
            "-R",  # Capital R -> follow symlinks
            name,
            ".",
        ],
        capture_output=True,
        text=True,
        cwd=output_dir,
        check=True,
    )
    outputs = p.stdout.splitlines()
    if len(outputs) == 0:
        raise FileNotFoundError(name)
    if len(outputs) > 1:
        raise RuntimeError(f"Multiple matches {name}: {outputs}")
    return Path(outputs[0]).parent
