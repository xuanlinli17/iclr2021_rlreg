# Regularizations in Policy Optimization

This repository contains the code for:  

Regularization Matters for Policy Optimization [[arXiv]](https://arxiv.org/abs/1910.09191). Also in NeurIPS 2019 Deep RL Workshop.

[Zhuang Liu](https://liuzhuang13.github.io/)\*, Xuanlin Li\*, [Bingyi Kang](scholar.google.com.sg/citations?user=NmHgX-wAAAAJ)\* and [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution)

Our code is adopted from [OpenAI Baselines](https://github.com/openai/baselines) and [SAC](https://github.com/haarnoja/sac).

## Abstract 
Deep Reinforcement Learning (Deep RL) has been receiving increasingly more
attention thanks to its encouraging performance on a variety of control tasks.
Yet, conventional regularization techniques in training neural networks (e.g.,
L<sub>2</sub> regularization, dropout) have been largely ignored in RL methods,
possibly because agents are typically trained and evaluated in the same
environment. In this work, we present the first comprehensive study of
regularization techniques with multiple policy optimization algorithms on
continuous control tasks. Interestingly, we find conventional regularization
techniques on the policy networks can often bring large improvement on the task
performance, and the improvement is typically more significant when the task is
more difficult. We also compare with the widely used entropy regularization and
find L<sub>2</sub> regularization is generally better. Our findings are further
confirmed to be robust against the choice of training hyperparameters. We also
study the effects of regularizing different components and find that only
regularizing the policy network is typically enough. We hope our study provides
guidance for future practices in regularizing policy optimization algorithms.


## Installation Instructions

Set up virtual environment using `virtualenv ENV_NAME --python=python3`

Install `mujoco_py` for `MuJoCo (version 2.0)` by following the instructions on https://github.com/openai/mujoco-py

Next, modify `.bashrc` (or set up a shell script before training):
```
export PYTHONPATH=PATH_TO_FOLDER/baselines_release:$PYTHONPATH
export PYTHONPATH=PATH_TO_FOLDER/rllab:$PYTHONPATH
export PYTHONPATH=PATH_TO_FOLDER/sac_release:$PYTHONPATH
```

Next, install the required packages. Openai baseline also requires that CUDA>=9.0.
```
pip3 install tensorflow-gpu==(VERSION_THAT_COMPLIES_WITH_CUDA_INSTALLATION)
pip3 install mpi4py roboschool==1.0.48 gym==0.13.0 click dill joblib opencv-python progressbar2 tqdm theano path.py cached_property python-dateutil pyopengl mako gtimer matplotlib pyprind
pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Running

To train, run 
```
python -m baselines.run --help
python sac_release/examples/mujoco_all_sac.py --help
```
for the available arguments, such as the number of environments simulated in parallel, model save path, etc.

### Regularization Options
```
l1regpi, l1regvf = L1 policy/value network regularization

l2regpi, l2regvf = L2 policy/value network regularization

wclippi, wclipvf = Policy/value network weight clipping
(Note: for openai baseline policy weight clipping, we only clip the mlp part of 
the network because clipping the log standard deviation vector almost always 
harms the performance)

dropoutpi, dropoutvf = Policy/value network dropout KEEP_PROB (1.0 = no dropout)

batchnormpi, batchnormvf = Policy/value network batch normalization (True or False)

ent_coef = Entropy regularization coefficient
```

### Examples:
```
python -m baselines.run --alg=ppo2 --env=RoboschoolHumanoid-v1 --num_timesteps=5e7 --l2regpi=0.0001
```
Runs `ppo2` (Proximal Policy Gradient) on `RoboschoolHumanoid` task with `5e7` timesteps with L2 regularization applied to the policy network with strength=0.0001.

```
python -m baselines.run --alg=a2c --env=Humanoid-v2 --num_timesteps=2e7 --ent_coef=0.0 --batchnormpi=True
```
Runs `a2c` (Synchronous version of A3C) on `Humanoid (MuJoCo)` task with `2e7` timesteps with batch normalization applied to the policy network and the entropy regularization turned off.

```
python sac_release/examples/mujoco_all_sac.py --env=atlas-forward-walk-roboschool --dropoutpi=0.9
```
Runs `sac` (Soft Actor Critic) on `RoboschoolAtlasForwardWalk` task with dropout probability = 1 - 0.9 = 0.1 on policy network (i.e. keep probability = 0.9).
(Note that the number of training timesteps is predefined in `sac_release/examples/variant.py`)


### Citation

```
@article{liu2019regularization,
  title={Regularization Matters for Policy Optimization},
  author={Liu, Zhuang and Li, Xuanlin and Kang, Bingyi and Darrell, Trevor},
  journal={arXiv preprint arXiv:1910.09191},
  year={2019}
}
```
