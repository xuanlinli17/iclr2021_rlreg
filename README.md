## Regularization Matters for Policy Optimization

### Installation instructions

First, install `mujoco_py` for `MuJoCo - 2.0` by following the instructions on https://github.com/openai/mujoco-py

Next, modify `.bashrc` (or set up a shell script before training):
```
export PYTHONPATH=PATH_TO_FOLDER/baselines_release:$PYTHONPATH
export PYTHONPATH=PATH_TO_FOLDER/rllab:$PYTHONPATH
export PYTHONPATH=PATH_TO_FOLDER/sac_release:$PYTHONPATH
```

Next, install the required packages. Openai baseline also requires that CUDA>=9.0.
```
pip3 install tensorflow-gpu==1.14.0 mpi4py roboschool==1.0.48 gym==0.13.0 click dill joblib opencv-python progressbar2 tqdm theano path.py cached_property python-dateutil pyopengl mako gtimer matplotlib pyprind
pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

### Glossary of Command Line Arguments (usage):

To train, run 
```
python -m baselines.run --help
python sac_release/examples/mujoco_all_sac.py --help
```
for the available arguments.

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

Examples:
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
