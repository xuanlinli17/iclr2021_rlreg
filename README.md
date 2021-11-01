# Regularizations in Policy Optimization - An Empirical Study on Continuous Control

This repository contains the code for:  

Regularization Matters for Policy Optimization - An Empirical Study on Continuous Control [[arXiv]](https://arxiv.org/abs/1910.09191). Also in ICLR 2021 and NeurIPS 2019 Deep RL Workshop.

[Zhuang Liu](https://liuzhuang13.github.io/)\*, Xuanlin Li\*, [Bingyi Kang](scholar.google.com.sg/citations?user=NmHgX-wAAAAJ)\* and [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (\* equal contribution)

Our code is adopted from [OpenAI Baselines](https://github.com/openai/baselines) and [SAC](https://github.com/haarnoja/sac).

## Abstract 
Deep Reinforcement Learning (Deep RL) has been receiving increasingly more attention  thanks to its encouraging performance on a variety of control tasks. Yet, conventional regularization techniques in training neural networks (e.g., $L_2$ regularization, dropout) have been largely ignored in RL methods, possibly because agents are typically trained and evaluated in the same environment, and because the deep RL community focuses more on high-level algorithm designs. In this work, we present the first comprehensive study of regularization techniques with multiple policy optimization algorithms on continuous control tasks. Interestingly, we find conventional regularization techniques on the policy networks can often bring large improvement, especially on harder tasks. Our findings are shown to be robust against training hyperparameter variations. We also compare these techniques with the more widely used entropy regularization. In addition, we study regularizing different components and find that only regularizing the policy network is typically the best. We further analyze why regularization may help generalization in RL from four perspectives - sample complexity, reward distribution, weight norm, and noise robustness. We hope our study provides guidance for future practices in regularizing policy optimization algorithms. 


## Installation Instructions

Git clone https://github.com/rll/rllab to PATH_TO_RLLAB_FOLDER

Install MuJoCo (but don't install `mujoco_py` yet) by following instructions on https://github.com/openai/mujoco-py.

Copy `additional_lib_for_rllab/libglfw.so.3, additional_lib_for_rllab/libmujoco131.so` in this repository and `mjkey.txt` from the mujoco key path to a new folder named `PATH_TO_RLLAB_FOLDER/vendor/mujoco`.

Fix a typo in rllab by
```
vi PATH_TO_RLLAB_FOLDER/rllab/sampler/stateful_pool.py
Change 
"from joblib.pool import MemmapingPool"
to
"from joblib.pool import MemmappingPool"
```

Set up virtual environment using 
```
virtualenv ENV_NAME --python=python3
source ENV_NAME/bin/activate
```

Install `mujoco_py` for `MuJoCo (version 2.0)` by following the instructions on https://github.com/openai/mujoco-py

Next, modify `.bashrc` (or set up a shell script named `SOMESCRIPT.sh` and `source SOMESCRIPT.sh` before training):
```
export PYTHONPATH=PATH_TO_THIS_REPO/baselines_release:$PYTHONPATH
export PYTHONPATH=PATH_TO_RLLAB_FOLDER:$PYTHONPATH
export PYTHONPATH=PATH_TO_THIS_REPO/sac_release:$PYTHONPATH
```

Next, install the required packages. Openai baseline also requires that CUDA>=9.0.
```
pip3 install tensorflow-gpu==(VERSION_THAT_COMPLIES_WITH_CUDA_INSTALLATION, note that tensorflow 2.0.0 is not compatible with this repo)
pip3 install mpi4py roboschool==1.0.48 gym==0.13.0 click dill joblib opencv-python progressbar2 tqdm theano path.py cached_property python-dateutil pyopengl mako gtimer matplotlib pyprind
pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Running

To train, run 
```
cd PATH_TO_THIS_REPO
python -m baselines.run --help
python PATH_TO_REPO/sac_release/examples/mujoco_all_sac.py --help
```
for the available arguments, such as the number of environments simulated in parallel, model save path, etc.

For Soft Actor Critic, `PATH_TO_THIS_REPO/sac_release/examples.variants.py` contains default environment settings. These settings are overwritten by command line arguments.

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
@inproceedings{liu2019regularization,
  title={Regularization Matters for Policy Optimization - An Empirical Study on Continuous Control},
  author={Liu, Zhuang and Li, Xuanlin and Kang, Bingyi and Darrell, Trevor},
  booktitle={International Conference on Learning Representations},
  year={2021}  
}
```
