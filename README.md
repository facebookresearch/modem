# MoDem: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations

Original PyTorch implementation of [MoDem: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations](#) by

[Nicklas Hansen](https://nicklashansen.github.io/), [Yixin Lin](https://yixinlin.net/), [Hao Su](https://cseweb.ucsd.edu/~haosu/), [Xiaolong Wang](https://xiaolonw.github.io/), [Vikash Kumar](https://vikashplus.github.io/), [Aravind Rajeswaran](https://aravindr93.github.io/) (Meta AI, UC San Diego)

<p align="center">
  <img width="19.5%" src="https://i.imgur.com/9zGHWnd.gif">
  <img width="19.5%" src="https://i.imgur.com/P4y8x6v.gif">
  <img width="19.5%" src="https://i.imgur.com/6gswKaN.gif">
  <img width="19.5%" src="https://i.imgur.com/XyGakRx.gif">
  <img width="19.5%" src="https://i.imgur.com/ewoSmCk.gif">
   <a href="#">[Paper]</a>&emsp;<a href="https://nicklashansen.github.io/modemrl">[Website]</a>
</p>


## Method

Our model-based method, **MoDem**, solves challenging visuo-motor control tasks with sparse rewards and high-dimensional action spaces in **100K** interaction steps given only **5** demonstrations.

<p align="center">
  <img width="80%" src="https://i.imgur.com/dsdDsSl.png">
</p>


## Citation

If you use this repo in your research, please consider citing the paper as follows:

```
@article{hansen2022modem,
  title={MoDem: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations},
  author={Nicklas Hansen and Yixin Lin and Hao Su and Xiaolong Wang and Vikash Kumar and Aravind Rajeswaran},
  journal={arXiv preprint},
  year={2022}
}
```


## Instructions

We assume that your machine has a CUDA-enabled GPU, a local copy of MuJoCo 2.1.x installed (required for the Adroit/Meta-World domains), and at least 80GB of memory. Then, create a conda environment with `conda env create -f environment.yml`, and add `/<path>/<to>/<your>/modem/tasks/mj_envs` to your `PYTHONPATH` (required for the Adroit domain). No additional setup required for the DMControl domain. You will also need to configure `wandb` and your demonstration/logging directories in `cfgs/config.yaml`. Demonstrations are made available [here](https://github.com/facebookresearch/modem/releases/tag/v.0.1.0). Once setup is complete, you should be able to run the following commands.

To train MoDem on a task from **Adroit**:

```
python train.py suite=adroit task=adroit-door
```

To train MoDem on a task from **Meta-World**:

```
python train.py suite=mw task=mw-assembly
```

To train MoDem on a task from **DMControl**:

```
python train.py suite=dmcontrol task=quadruped-run
```


## License & Acknowledgements

This codebase is based on the original [TD-MPC](https://github.com/nicklashansen/tdmpc) implementation. MoDem, TD-MPC and [Meta-World](https://github.com/rlworkgroup/metaworld) are licensed under the MIT license. [MuJoCo](https://github.com/deepmind/mujoco), [DeepMind Control Suite](https://github.com/deepmind/dm_control), and [mj_envs](https://github.com/vikashplus/mj_envs) (Adroit) are licensed under the Apache 2.0 license. We thank the [DrQv2](https://github.com/facebookresearch/drqv2) authors for their implementation of DMControl wrappers.
