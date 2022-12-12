# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import numpy as np
import gym
from gym.wrappers import TimeLimit
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._num_frames * 3, cfg.img_size, cfg.img_size),
            dtype=np.uint8,
        )
        self.action_space = self.env.action_space
        self.camera_name = "corner2"
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]

    @property
    def state(self):
        state = self._state_obs.astype(np.float32)
        return np.concatenate((state[:4], state[18 : 18 + 4]))

    def _get_pixel_obs(self):
        return self.render(width=self.cfg.img_size, height=self.cfg.img_size).transpose(
            2, 0, 1
        )

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def reset(self):
        self.env.reset()
        obs = self.env.step(np.zeros_like(self.env.action_space.sample()))[0].astype(
            np.float32
        )
        self._state_obs = obs
        obs = self._get_pixel_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()

    def step(self, action):
        reward = 0
        for _ in range(self.cfg.action_repeat):
            obs, r, _, info = self.env.step(action)
            reward += r
        obs = obs.astype(np.float32)
        self._state_obs = obs
        obs = self._get_pixel_obs()
        self._frames.append(obs)
        reward = float(info["success"]) - 1.0
        return self._stacked_obs(), reward, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        return self.env.render(
            offscreen=True, resolution=(width, height), camera_name=self.camera_name
        ).copy()

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_metaworld_env(cfg):
    env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
    env._freeze_rand_vec = False
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    cfg.state_dim = 8
    return env
