#!/usr/bin/env python

import hydra
from omegaconf import DictConfig

# Keep the imports to a minimum so that hydra --cfg doesn't take ages


@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    import osc.train

    osc.train.main(cfg)


if __name__ == "__main__":
    main()
