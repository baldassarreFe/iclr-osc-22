import logging
from pathlib import Path

import hydra
import tabulate
from omegaconf import DictConfig, OmegaConf

from osc.train import (
    build_dataset_train,
    build_dataset_val,
    build_losses,
    build_model,
    build_optimizer,
    build_scheduler,
    get_viz_batch,
    run_train_val_viz_epochs,
    update_cfg,
)
from osc.utils import seed_everything
from osc.wandb_utils import setup_wandb

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    log.info("Config original:\n%s", OmegaConf.to_yaml(cfg))
    cfg = update_cfg(cfg)
    with open("train.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.other.seed)
    setup_wandb(cfg)

    ds_train = build_dataset_train(cfg)
    ds_val = build_dataset_val(cfg)
    viz_batch = get_viz_batch(cfg)

    model = build_model(cfg).to(cfg.other.device)
    log.info(
        "Model parameters:\n%s",
        tabulate.tabulate(
            [
                [name, sum(p.numel() for p in child.parameters() if p.requires_grad)]
                for name, child in model.named_children()
            ],
            headers=["Module", "Parameters"],
        ),
    )

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    loss_fn_global, loss_fn_objects = build_losses(cfg)
    run_train_val_viz_epochs(
        cfg,
        model,
        optimizer,
        scheduler,
        ds_train,
        ds_val,
        viz_batch,
        loss_fn_global,
        loss_fn_objects,
    )

    log.info("Run dir: %s", Path.cwd().relative_to(hydra.utils.get_original_cwd()))


if __name__ == "__main__":
    main()
