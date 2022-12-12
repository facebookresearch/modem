# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import torch
import numpy as np
import gym
from gym.wrappers import TimeLimit
import mj_envs.envs.hand_manipulation_suite


class AdroitWrapper(gym.Wrapper):
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
        self.camera_name = cfg.get("camera_view", "view_1")

    @property
    def state(self):
        return self._state_obs.astype(np.float32)

    def _get_state_obs(self, obs):
        if self.cfg.task == "adroit-door":
            qp = self.env.data.qpos.ravel()
            palm_pos = self.env.data.site_xpos[self.env.grasp_sid].ravel()
            manual = np.concatenate([qp[1:-2], palm_pos])
            obs = np.concatenate([obs[:27], obs[29:32]])
            assert np.isclose(obs, manual).all()
            return obs
        elif self.cfg.task == "adroit-hammer":
            qp = self.env.data.qpos.ravel()
            qv = np.clip(self.env.data.qvel.ravel(), -1.0, 1.0)
            palm_pos = self.env.data.site_xpos[self.env.S_grasp_sid].ravel()
            manual = np.concatenate([qp[:-6], qv[-6:], palm_pos])
            obs = obs[:36]
            assert np.isclose(obs, manual).all()
            return obs
        elif self.cfg.task == "adroit-pen":
            qp = self.env.data.qpos.ravel()
            desired_orien = (
                self.env.data.site_xpos[self.env.tar_t_sid]
                - self.env.data.site_xpos[self.env.tar_b_sid]
            ) / self.env.tar_length
            manual = np.concatenate([qp[:-6], desired_orien])
            obs = np.concatenate([obs[:24], obs[-9:-6]])
            assert np.isclose(obs, manual).all()
            return obs
        elif self.cfg.task == "adroit-relocate":
            qp = self.env.data.qpos.ravel()
            palm_pos = self.env.data.site_xpos[self.env.S_grasp_sid].ravel()
            target_pos = self.env.data.site_xpos[self.env.target_obj_sid].ravel()
            manual = np.concatenate([qp[:-6], palm_pos - target_pos])
            obs = np.concatenate([obs[:30], obs[-6:-3]])
            assert np.isclose(obs, manual).all()
            return obs
        raise NotImplementedError()

    def _get_pixel_obs(self):
        return self.render(width=self.cfg.img_size, height=self.cfg.img_size).transpose(
            2, 0, 1
        )

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=0)

    def reset(self):
        obs = self.env.reset()
        self._state_obs = self._get_state_obs(obs)
        obs = self._get_pixel_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()

    def step(self, action):
        reward = 0
        for _ in range(self.cfg.action_repeat):
            obs, r, _, info = self.env.step(action)
            reward += r
        self._state_obs = self._get_state_obs(obs)
        obs = self._get_pixel_obs()
        self._frames.append(obs)
        info["success"] = info["goal_achieved"]
        reward = float(info["success"]) - 1.0
        return self._stacked_obs(), reward, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        return np.flip(
            self.env.env.sim.render(
                mode="offscreen",
                width=width,
                height=height,
                camera_name=self.camera_name,
            ),
            axis=0,
        )

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_adroit_env(cfg):
    env_id = cfg.task.split("-", 1)[-1] + "-v0"
    env = gym.make(env_id)
    env = AdroitWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    env.reset()
    cfg.state_dim = env.state.shape[0]
    return env
