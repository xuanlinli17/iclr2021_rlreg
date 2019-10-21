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

#### Glossary:

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