# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import numpy as np
import torch
import random
import gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DefaultDictWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.success = False

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        self.success = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = defaultdict(float, info)
        self.success = self.success or bool(info["success"])
        info["success"] = float(self.success)
        return obs, reward, done, info


def make_env(cfg):
    """
    Make environment for experiments.
    """
    domain, _ = cfg.task.split("-", 1)
    if domain == "mw":  # Meta-World
        from tasks.metaworld import make_metaworld_env

        env = make_metaworld_env(cfg)
    elif domain == "adroit":  # Adroit
        from tasks.adroit import make_adroit_env

        env = make_adroit_env(cfg)
    else:  # DMControl
        from tasks.dmcontrol import make_dmcontrol_env

        env = make_dmcontrol_env(cfg)

    env = DefaultDictWrapper(env)
    cfg.domain = domain
    cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    cfg.action_shape = tuple(int(x) for x in env.action_space.shape)
    cfg.action_dim = env.action_space.shape[0]
    return env
