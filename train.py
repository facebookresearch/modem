# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, get_demos, ReplayBuffer
from termcolor import colored
from copy import deepcopy
import logger
import hydra
torch.backends.cudnn.benchmark = True


def evaluate(env, agent, cfg, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	episode_successes = []
	for i in range(cfg.eval_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, env.state, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, info = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		episode_successes.append(info.get('success', 0))
		if video: video.save(env_step)
	return np.nanmean(episode_rewards), np.nanmean(episode_successes)


@hydra.main(config_name='config', config_path='cfgs')
def train(cfg: dict):
	"""Training script for online TD-MPC."""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	work_dir = Path(cfg.logging_dir) / 'logs' / cfg.task / cfg.exp_name / str(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), work_dir)
	env, agent = make_env(cfg), TDMPC(cfg)
	demo_buffer = ReplayBuffer(deepcopy(cfg)) if cfg.get('demos', 0) > 0 else None
	buffer = ReplayBuffer(cfg)
	L = logger.Logger(work_dir, cfg)
	print(agent.model)

	# Load demonstrations
	if cfg.get('demos', 0) > 0:
		for episode in get_demos(cfg):
			demo_buffer += episode
		print(colored(f'Loaded {cfg.demos} demonstrations', 'yellow', attrs=['bold']))
		print(colored('Phase 1: policy pretraining', 'red', attrs=['bold']))
		agent.init_bc(demo_buffer if cfg.get('demo_schedule', 0) != 0 else buffer)
		print(colored('\nPhase 2: seeding', 'green', attrs=['bold']))

	# Run training
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs, env.state)
		while not episode.done:
			action = agent.plan(obs, env.state, step=step, t0=episode.first)
			obs, reward, done, info = env.step(action.cpu().numpy())
			episode += (obs, env.state, action, reward, done)
		assert len(episode) == cfg.episode_length, f'Episode length {len(episode)} != {cfg.episode_length}'
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			if step == cfg.seed_steps:
				print(colored('Seeding complete, pretraining model...', 'yellow', attrs=['bold']))
				num_updates = cfg.seed_steps
			else:
				num_updates = cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i, demo_buffer))
			if step == cfg.seed_steps:
				print(colored('Phase 3: interactive learning', 'blue', attrs=['bold']))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward,
			'episode_success': info.get('success', 0)}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			eval_rew, eval_succ = evaluate(env, agent, cfg, step, env_step, L.video)
			common_metrics.update({
				'episode_reward': eval_rew,
				'episode_success': eval_succ})
			L.log(common_metrics, category='eval')
			if cfg.save_model and env_step % cfg.save_freq == 0 and env_step > 0:
				L.save_model(agent, env_step)
				print(colored(f'Model has been checkpointed', 'yellow', attrs=['bold']))

	L.finish()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
