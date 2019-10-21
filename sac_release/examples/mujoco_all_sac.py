import argparse
import os

import tensorflow as tf
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
# from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sac.algos import SAC
from sac.envs import (
    GymEnv,
    MultiDirectionSwimmerEnv,
    MultiDirectionAntEnv,
    MultiDirectionHumanoidEnv,
    CrossMazeAntEnv,
)

from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp, unflatten
from sac.policies import GaussianPolicy, LatentSpacePolicy, GMMPolicy, UniformPolicy
from sac.misc.sampler import SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.preprocessors import MLPPreprocessor
from examples.variants import parse_domain_and_task, get_variants
import copy

ENVIRONMENTS = {
    'swimmer-gym': {
        'default': lambda: GymEnv('Swimmer-v2'),
    },
    'swimmer-rllab': {
        'default': SwimmerEnv,
        'multi-direction': MultiDirectionSwimmerEnv,
    },
    'ant': {
        'default': lambda: GymEnv('Ant-v2'),
        'multi-direction': MultiDirectionAntEnv,
        'cross-maze': CrossMazeAntEnv
    },
    'humanoid-gym': {
        'default': lambda: GymEnv('Humanoid-v2')
    },
    'humanoid-rllab': {
        'default': HumanoidEnv,
        'multi-direction': MultiDirectionHumanoidEnv,
    },
    'hopper': {
        'default': lambda: GymEnv('Hopper-v2')
    },
    'half-cheetah': {
        'default': lambda: GymEnv('HalfCheetah-v2')
    },
    'walker': {
        'default': lambda: GymEnv('Walker2d-v2')
    },
    'humanoid-standup-gym': {
        'default': lambda: GymEnv('HumanoidStandup-v2')
    },
    'humanoid-roboschool': {
        'default': lambda: GymEnv('RoboschoolHumanoid-v1')
    },
    'atlas-forward-walk-roboschool': {
        'default': lambda: GymEnv('RoboschoolAtlasForwardWalk-v1')
    },
    'humanoid-flagrun-roboschool': {
        'default': lambda: GymEnv('RoboschoolHumanoidFlagrun-v1')
    },
    'humanoid-flagrun-harder-roboschool': {
        'default': lambda: GymEnv('RoboschoolHumanoidFlagrunHarder-v1')
    }
}

DEFAULT_DOMAIN = DEFAULT_ENV = 'swimmer-rllab'
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())
AVAILABLE_TASKS = set(y for x in ENVIRONMENTS.values() for y in x.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default=None)
    parser.add_argument('--task',
                        type=str,
                        choices=AVAILABLE_TASKS,
                        default='default')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian', 'gmm', 'lsp'),
                        default='gaussian')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--lr', type=float, default=-1.0)
    parser.add_argument('--l1regpi', type=float, default=0.0) #L1 reg policy
    parser.add_argument('--l2regpi', type=float, default=0.0) #L2 reg policy
    parser.add_argument('--l1regvf', type=float, default=0.0) #L1 reg value (V only, the two Q networks are not regularized for simplicity)
    parser.add_argument('--l2regvf', type=float, default=0.0) #L2 reg value
    parser.add_argument('--wclippi', type=float, default=0.0) #Weight clip policy
    parser.add_argument('--wclipvf', type=float, default=0.0) #Weight clip value
    parser.add_argument('--dropoutpi', type=float, default=1.0) #Dropout policy keep prob
    parser.add_argument('--dropoutvf', type=float, default=1.0) #Dropout value keep prob
    parser.add_argument('--ent_coef', type=float, default=0.0) #Entropy regularization coefficient, not the temperature of maximum entropy formulation (In this SAC implementation, this temperature term is fixed to be 1.
    parser.add_argument('--batchnormpi', type=bool, default=False) #Batchnorm policy (T/F)
    parser.add_argument('--batchnormvf', type=bool, default=False) #Batchnorm value (T/F)
    #parser.add_argument('--gaussianreg', type=float, default=1e-3) #This term appears in the original code release and regularizes the mu and logsigma of policy output. However, this regularization term is not applied when we use gaussian policy. Since all of our experiements adopt gaussian policy, this term does not affect our result in anyway. 
    parser.add_argument('--reward_scale', type=float, default=-1.0)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--policypath', type=str, default='') #Policy network save path
    parser.add_argument('--valuepath', type=str, default='') #Value network save path
    args = parser.parse_args()

    return args

def run_experiment(variant):
    env_params = variant['env_params']
    policy_params = variant['policy_params']
    value_fn_params = variant['value_fn_params']
    algorithm_params = variant['algorithm_params']
    replay_buffer_params = variant['replay_buffer_params']
    sampler_params = variant['sampler_params']

    task = variant['task']
    domain = variant['domain']

    env = normalize(ENVIRONMENTS[domain][task](**env_params))

    pool = SimpleReplayBuffer(env_spec=env.spec, **replay_buffer_params)

    sampler = SimpleSampler(**sampler_params)

    base_kwargs = dict(algorithm_params['base_kwargs'], sampler=sampler)

    M = value_fn_params['layer_size']
    if variant['num_hidden'] != 256:
        M = variant['num_hidden']
    qf1 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf1', batchnormvf=variant['batchnormvf'])
    qf2 = NNQFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), name='qf2', batchnormvf=variant['batchnormvf'])
    vf = NNVFunction(env_spec=env.spec, hidden_layer_sizes=(M, M), batchnormvf=variant['batchnormvf'], dropoutvf_keep_prob=variant['dropoutvf'])

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)

    if policy_params['type'] == 'gaussian':
        policy = GaussianPolicy(
                env_spec=env.spec,
                hidden_layer_sizes=(M,M),
                reparameterize=policy_params['reparameterize'],
                todropoutpi=(variant['dropoutpi']<1.0),
                dropoutpi=variant['dropoutpi'],
                batchnormpi=variant['batchnormpi']
        )
    elif policy_params['type'] == 'lsp':
        nonlinearity = {
            None: None,
            'relu': tf.nn.relu,
            'tanh': tf.nn.tanh
        }[policy_params['preprocessing_output_nonlinearity']]

        preprocessing_hidden_sizes = policy_params.get('preprocessing_hidden_sizes')
        if preprocessing_hidden_sizes is not None:
            observations_preprocessor = MLPPreprocessor(
                env_spec=env.spec,
                layer_sizes=preprocessing_hidden_sizes,
                output_nonlinearity=nonlinearity)
        else:
            observations_preprocessor = None

        policy_s_t_layers = policy_params['s_t_layers']
        policy_s_t_units = policy_params['s_t_units']
        s_t_hidden_sizes = [policy_s_t_units] * policy_s_t_layers

        bijector_config = {
            'num_coupling_layers': policy_params['coupling_layers'],
            'translation_hidden_sizes': s_t_hidden_sizes,
            'scale_hidden_sizes': s_t_hidden_sizes,
        }

        policy = LatentSpacePolicy(
            env_spec=env.spec,
            squash=policy_params['squash'],
            bijector_config=bijector_config,
            reparameterize=policy_params['reparameterize'],
            q_function=qf1,
            observations_preprocessor=observations_preprocessor)
    elif policy_params['type'] == 'gmm':
        # reparameterize should always be False if using a GMMPolicy
        policy = GMMPolicy(
            env_spec=env.spec,
            K=policy_params['K'],
            hidden_layer_sizes=(M, M),
            reparameterize=policy_params['reparameterize'],
            qf=qf1,
            reg=1e-3,
        )
    else:
        raise NotImplementedError(policy_params['type'])

    if variant['reward_scale'] < 0:
        scale_rew = algorithm_params['scale_reward']
    else:
        scale_rew = variant['reward_scale']
    algorithm = SAC(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        initial_exploration_policy=initial_exploration_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        lr=algorithm_params['lr'] if variant['lr'] < 0 else variant['lr'],
        scale_reward=scale_rew,
        discount=algorithm_params['discount'],
        tau=variant['tau'],
        reparameterize=algorithm_params['reparameterize'],
        target_update_interval=algorithm_params['target_update_interval'],
        action_prior=policy_params['action_prior'],
        save_full_state=False,
        l1regpi=variant['l1regpi'],
        l2regpi=variant['l2regpi'],
        l1regvf=variant['l1regvf'],
        l2regvf=variant['l2regvf'],
        ent_coef=variant['ent_coef'],
        wclippi=variant['wclippi'],
        wclipvf=variant['wclipvf'],
        dropoutpi=variant['dropoutpi'],
        dropoutvf=variant['dropoutvf'],
        batchnormpi=variant['batchnormpi'],
        batchnormvf=variant['batchnormvf']
    )

    algorithm._sess.run(tf.global_variables_initializer())

    for v in tf.trainable_variables():
        print(v.name)

    algorithm.train()

    if variant['policypath'] != '':
        save_w_path = os.path.expanduser(variant['policypath'])
        toexport = []
        savesess = algorithm._sess
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gaussian_policy'):
            toexport.append(savesess.run(v))
        np.savetxt(save_w_path, np.concatenate(toexport, axis=None), delimiter=',')        
    if variant['valuepath'] != '':
        save_w_path = os.path.expanduser(variant['valuepath'])
        toexport = []
        savesess = algorithm._sess
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qf1'):
            toexport.append(savesess.run(v))
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qf2'):
            toexport.append(savesess.run(v))
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vf'):
            toexport.append(savesess.run(v))
        np.savetxt(save_w_path, np.concatenate(toexport, axis=None), delimiter=',')
 


def launch_experiments(variant_generator, args):
    variants = variant_generator.variants()
    # TODO: Remove unflatten. Our variant generator should support nested params
    variants = [unflatten(variant, separator='.') for variant in variants]

    print('Launching seed={} experiment.'.format(args.seed))
    variant = variants[args.seed - 1]
    variant['lr'] = args.lr
    variant['tau'] = args.tau
    variant['l1regpi'] = args.l1regpi
    variant['l2regpi'] = args.l2regpi
    variant['l1regvf'] = args.l1regvf
    variant['l2regvf'] = args.l2regvf
    variant['wclippi'] = args.wclippi
    variant['wclipvf'] = args.wclipvf
    variant['dropoutpi'] = args.dropoutpi
    variant['dropoutvf'] = args.dropoutvf
    variant['ent_coef'] = args.ent_coef
    variant['batchnormpi'] = args.batchnormpi
    variant['batchnormvf'] = args.batchnormvf
    variant['reward_scale'] = args.reward_scale
    variant['num_hidden'] = args.num_hidden
    variant['policypath'] = args.policypath
    variant['valuepath'] = args.valuepath

    print("Variant for this experiment:", variant)
    run_params = variant['run_params']
    algo_params = variant['algorithm_params']

    experiment_prefix = variant['prefix'] + '/' + args.exp_name
    experiment_name = '{prefix}-{exp_name}-{i:02}'.format(
        prefix=variant['prefix'], exp_name=args.exp_name, i=args.seed)

    run_sac_experiment(
        run_experiment,
        mode=args.mode,
        variant=variant,
        exp_prefix=experiment_prefix,
        exp_name=experiment_name,
        n_parallel=1,
        seed=run_params['seed'],
        terminate_machine=True,
        log_dir=args.log_dir,
        snapshot_mode=run_params['snapshot_mode'],
        snapshot_gap=run_params['snapshot_gap'],
        sync_s3_pkl=run_params['sync_pkl'],
    )


def main():
    tf.enable_eager_execution()
    args = parse_args()
    print(__name__)
    domain, task = args.domain, args.task
    if (not domain) or (not task):
        domain, task = parse_domain_and_task(args.env)

    variant_generator = get_variants(domain=domain, task=task, policy=args.policy)
    launch_experiments(variant_generator, args)


if __name__ == '__main__':
    main()
