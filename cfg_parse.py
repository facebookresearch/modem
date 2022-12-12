# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
from omegaconf import OmegaConf


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config file.
    Adapted from https://github.com/nicklashansen/tdmpc
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass

    # Convenience
    cfg.task_title = cfg.task.replace("-", " ").title()

    # Overwrite task-specific episode lengths (after action repeat)
    if cfg.task.startswith("adroit-"):
        if cfg.task == "adroit-hammer":
            cfg.episode_length = 125
        elif cfg.task == "adroit-pen":
            cfg.episode_length = 50
        elif cfg.task == "adroit-door":
            cfg.episode_length = 100
        else:
            raise ValueError(f'Invalid Adroit task "{cfg.task}"')

    return cfg
