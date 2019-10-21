import time
import functools
import tensorflow as tf

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.a2c.utils import Scheduler, find_trainable_variables
from baselines.a2c.runner import Runner

from tensorflow import losses
class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps, dropoutpi_keep_prob, dropoutpi_keep_prob_value,
            dropoutvf_keep_prob, dropoutvf_keep_prob_value, isbnpitrainmode, isbnvftrainmode,
            l1regpi, l2regpi, l1regvf, l2regvf, wclippi, wclipvf,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            regnologstd=False, regonlylogstd=False):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])
        
        self.dropoutpi_keep_prob = dropoutpi_keep_prob
        self.dropoutpi_keep_prob_value = dropoutpi_keep_prob_value
        self.dropoutvf_keep_prob = dropoutvf_keep_prob
        self.dropoutvf_keep_prob_value = dropoutvf_keep_prob_value
        self.isbnpitrainmode = isbnpitrainmode
        self.isbnvftrainmode = isbnvftrainmode

        #REGULARIZATION
        self.toregularizepi = l1regpi > 0 or l2regpi > 0
        self.toregularizevf = l1regvf > 0 or l2regvf > 0
        self.toweightclippi = wclippi > 0
        self.toweightclipvf = wclipvf > 0

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        if self.toregularizepi:
            print("Regularizing policy network: L1 = {}, L2 = {}".format(l1regpi, l2regpi))
            regularizerpi = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1regpi, scale_l2=l2regpi, scope='a2c_model/pi')
            all_trainable_weights_pi = find_trainable_variables('a2c_model/pi')
            regularization_penalty_pi = tf.contrib.layers.apply_regularization(regularizerpi, all_trainable_weights_pi)
            loss = loss + regularization_penalty_pi
        if self.toregularizevf:
            print("Regularizing value network: L1 = {}, L2 = {}".format(l1regvf, l2regvf))
            regularizervf = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1regvf, scale_l2=l2regvf, scope='a2c_model/vf')
            all_trainable_weights_vf = find_trainable_variables('a2c_model/vf')
            regularization_penalty_vf = tf.contrib.layers.apply_regularization(regularizervf, all_trainable_weights_vf)
            loss = loss + regularization_penalty_vf


        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        if self.toweightclippi:
             print("Weight clipping policy network = {}".format(wclippi))
             policyparams = find_trainable_variables('a2c_model/pi')
             self._wclip_ops_pi = []
             self.wclip_bounds_pi = [-wclippi, wclippi]
             for toclipvar in policyparams:
                 if 'logstd' in toclipvar.name:
                     continue
                 self._wclip_ops_pi.append(tf.assign(toclipvar, tf.clip_by_value(toclipvar, self.wclip_bounds_pi[0], self.wclip_bounds_pi[1])))
             self._wclip_op_pi = tf.group(*self._wclip_ops_pi)
        if self.toweightclipvf:
             print("Weight clipping value network = {}".format(wclipvf))
             valueparams = find_trainable_variables('a2c_model/vf')
             self._wclip_ops_vf = []
             self.wclip_bounds_vf = [-wclipvf, wclipvf]
             for toclipvar in valueparams:
                 self._wclip_ops_vf.append(tf.assign(toclipvar, tf.clip_by_value(toclipvar, self.wclip_bounds_vf[0], self.wclip_bounds_vf[1])))
             self._wclip_op_vf = tf.group(*self._wclip_ops_vf)


        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            if self.dropoutpi_keep_prob is not None:
                td_map[self.dropoutpi_keep_prob] = self.dropoutpi_keep_prob_value
            if self.dropoutvf_keep_prob is not None:
                td_map[self.dropoutvf_keep_prob] = self.dropoutvf_keep_prob_value
            if self.isbnpitrainmode is not None:
                td_map[self.isbnpitrainmode] = True
            if self.isbnvftrainmode is not None:
                td_map[self.isbnvftrainmode] = True
            train_tensors = [pg_loss, vf_loss, entropy, _train]
            if self.toweightclippi:
                train_tensors.append(self._wclip_op_pi)
            if self.toweightclipvf:
                train_tensors.append(self._wclip_op_vf)
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                train_tensors, td_map)[:4]
            return policy_loss, value_loss, policy_entropy


        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''



    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    todropoutpi = network_kwargs["dropoutpi"] < 1.0
    todropoutvf = network_kwargs["dropoutvf"] < 1.0
    batchnormpi = network_kwargs["batchnormpi"]
    batchnormvf = network_kwargs["batchnormvf"]
    if todropoutpi or todropoutvf:
        policy, dropoutpi_keep_prob, dropoutvf_keep_prob = build_policy(env, network, **network_kwargs)
    else:
        policy = build_policy(env, network, **network_kwargs)
    isbnpitrainmode = None
    isbnvftrainmode = None
    if batchnormpi and batchnormvf:
        policy, isbnpitrainmode, isbnvftrainmode = policy
    elif batchnormpi and not batchnormvf:
        policy, isbnpitrainmode = policy
    elif batchnormvf and not batchnormpi:
        policy, isbnvftrainmode = policy
    
    # Instantiate the model object (that creates step_model and train_model)
    model = Model(policy=policy, env=env, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
        l1regpi=network_kwargs["l1regpi"], l2regpi=network_kwargs["l2regpi"], l1regvf=network_kwargs["l1regvf"], l2regvf=network_kwargs["l2regvf"],
        wclippi=network_kwargs["wclippi"], wclipvf=network_kwargs["wclipvf"],
        lrschedule=lrschedule, dropoutpi_keep_prob=(dropoutpi_keep_prob if todropoutpi else None), dropoutpi_keep_prob_value=(network_kwargs['dropoutpi'] if todropoutpi else 1.0),
        dropoutvf_keep_prob=(dropoutvf_keep_prob if todropoutvf else None), dropoutvf_keep_prob_value=(network_kwargs['dropoutvf'] if todropoutvf else 1.0),
        isbnpitrainmode = isbnpitrainmode, isbnvftrainmode = isbnvftrainmode,)
    if load_path is not None:
        model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, 
            dropoutpi=(network_kwargs['dropoutpi'] if todropoutpi else 1.0), 
            dropoutvf=(network_kwargs['dropoutvf'] if todropoutvf else 1.0),
            isbnpitrainmode=isbnpitrainmode,
            isbnvftrainmode=isbnvftrainmode)

    # Calculate the batch_size
    nbatch = nenvs*nsteps

    # Start total timer
    tstart = time.time()

    for update in range(1, total_timesteps//nbatch+1):
        # Get mini batch of experiences
        obs, states, rewards, masks, actions, values = runner.run()

        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart

        # Calculate the fps (frame per second)
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    return model

