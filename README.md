## Regularization Matters for Policy Optimization

Modify `.bashrc` (or set up a shell script before training):
```
export PYTHONPATH=PATH_TO_FOLDER/baselines_release:$PYTHONPATH
export PYTHONPATH=PATH_TO_FOLDER/sac:$PYTHONPATH
```

To train, run 
```
python -m baselines.run --help
python sac_release/examples/mujoco_all_sac.py --help
```
for the available arguments.

#### Glossary of Command Line Arguments (usage):

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
Runs `ppo2` on `RoboschoolHumanoid` task with `5e7` timesteps with L2 regularization applied to the policy network with strength=0.0001.

```
python -m baselines.run --alg=a2c --env=Humanoid-v2 --num_timesteps=2e7 --ent_coef=0.0 --batchnormpi=True
```
Runs `a2c` on `Humanoid (MuJoCo)` task with `2e7` timesteps with batch normalization applied to the policy network and the entropy regularization turned off.

```
python sac_release/examples/mujoco_all_sac.py --env=atlas-forward-walk-roboschool --dropoutpi=0.9
```
Runs `sac` on `RoboschoolAtlasForwardWalk` task with dropout probability = 1 - 0.9 = 0.1 on policy network (i.e. keep probability = 0.9).
(Note that the number of training timesteps is predefined in `sac_release/examples/variant.py`)
