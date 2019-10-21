from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
from gym import spaces
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from contextlib import contextmanager

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def traj_segment_generator(pi, env, horizon, nenvs, stochastic, dropoutpi_keep_prob, dropoutvf_keep_prob, isbnpitrainmode, isbnvftrainmode):
    # Initialize state variables
    t = 0
    ac = [env.action_space.sample()] * nenvs
    new = [True] * nenvs
    rew = [0.0] * nenvs
    ob = env.reset()

    cur_ep_ret = []
    cur_ep_len = []
    ep_rets = []
    ep_lens = []

    for _ in range(nenvs):
        ep_rets.append([])
        ep_lens.append([])
        cur_ep_ret.append(0)
        cur_ep_len.append(0)

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros([horizon, nenvs], 'float32')
    vpreds = np.zeros([horizon, nenvs], 'float32')
    news = np.zeros([horizon, nenvs], 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        stepdict = {"stochastic": stochastic}
        if dropoutpi_keep_prob is not None:
            stepdict.update({"dropoutpi_keep_prob": 1.0})
        if dropoutvf_keep_prob is not None:
            stepdict.update({"dropoutvf_keep_prob": 1.0})
        if isbnpitrainmode is not None:
            stepdict.update({"isbnpitrainmode": False})
        if isbnvftrainmode is not None:
            stepdict.update({"isbnvftrainmode": False})
        
        ac, vpred, _, _ = pi.step(ob, **stepdict)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}

            _, vpred, _, _ = pi.step(ob, **stepdict)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            for _ in range(nenvs):
                ep_rets.append([])
                ep_lens.append([])

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        for j in range(nenvs):
            cur_ep_len[j] += 1
            if isinstance(rew, float): #if not using vecenv wrapper
                cur_ep_ret[j] += rew
                newj = new
            else:
                cur_ep_ret[j] += rew[j]
                newj = new[j]
            if newj:
                ep_rets[j].append(cur_ep_ret[j])
                ep_lens[j].append(cur_ep_len[j])
                cur_ep_ret[j] = 0
                cur_ep_len[j] = 0
                if isinstance(rew, float):
                    ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam, nenvs):
    new = np.vstack([seg["new"], np.zeros(nenvs)]) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.vstack([seg["vpred"], seg["nextvpred"]])
    T = seg["rew"].shape[0]
    seg["adv"] = gaelam = np.empty([T, nenvs], dtype='float32')
    rew = seg["rew"]
    lastgaelam = np.zeros([nenvs], dtype=np.float32)
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

    tmparr = [-1]
    tmparr.extend(list(seg["ob"].shape[2:]))
    seg["ob"] = seg["ob"].reshape(tmparr)
    seg["new"] = seg["new"].flatten()
    seg["vpred"] = seg["vpred"].flatten()
    seg["nextvpred"] = seg["nextvpred"].flatten()
    seg["rew"] = seg["rew"].flatten()
    seg["adv"] = seg["adv"].flatten()
    seg["tdlamret"] = seg["tdlamret"].flatten()

def learn(*,
        network,
        env,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        vf_batch_size=64,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''
    if not isinstance(env.action_space, spaces.Discrete): #Atari envs are already parallel without using vecnormalize wrapper
        nenvs = env.num_envs
    else:
        nenvs = 1
    nbatch = nenvs*timesteps_per_batch
    l1regpi = network_kwargs['l1regpi']
    l2regpi = network_kwargs['l2regpi']
    l1regvf = network_kwargs['l1regvf']
    l2regvf = network_kwargs['l2regvf']
    toregularizepi = l1regpi > 0 or l2regpi > 0
    toregularizevf = l1regvf > 0 or l2regvf > 0
    toweightclippi = False
    toweightclipvf = False
    weight_clip_range_pi = 0
    weight_clip_range_vf = 0
    if network_kwargs['wclippi'] > 0:
       weight_clip_range_pi = network_kwargs['wclippi']
       print("Clipping policy network = {}".format(weight_clip_range_pi))
       toweightclippi = True
    if network_kwargs['wclipvf'] > 0:
       weight_clip_range_vf = network_kwargs['wclipvf']
       print("Clipping value network = {}".format(weight_clip_range_vf))
       toweightclipvf = True

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))

    batchnormpi = network_kwargs["batchnormpi"]
    batchnormvf = network_kwargs["batchnormvf"]
    dropoutpi_keep_prob = None
    todropoutpi = network_kwargs["dropoutpi"] < 1.0
    dropoutvf_keep_prob = None
    todropoutvf = network_kwargs["dropoutvf"] < 1.0
    if todropoutpi or todropoutvf:
        policy, dropoutpi_keep_prob, dropoutvf_keep_prob = build_policy(env, network, value_network="copy", **network_kwargs)
    else:
        policy = build_policy(env, network, value_network="copy", **network_kwargs)
    
    isbnpitrainmode = None
    isbnvftrainmode = None
    if batchnormpi and batchnormvf:
        policy, isbnpitrainmode, isbnvftrainmode = policy
    elif batchnormpi and not batchnormvf:
        policy, isbnpitrainmode = policy
    elif batchnormvf and not batchnormpi:
        policy, isbnvftrainmode = policy
    #policy = build_policy(env, network, value_network='copy', **network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = ent_coef * meanent

    vferr = tf.reduce_mean(tf.square(pi.vf - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    if toregularizepi:
        print("Regularizing policy network: L1 = {}, L2 = {}".format(l1regpi, l2regpi))
        regularizerpi = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1regpi, scale_l2=l2regpi, scope="pi")
        all_trainable_weights_pi = var_list.copy()
        regularization_penalty_pi = tf.contrib.layers.apply_regularization(regularizerpi, all_trainable_weights_pi)
        optimgain = optimgain - regularization_penalty_pi
    
    if toregularizevf:
        print("Regularizing value network: L1 = {}, L2 = {}".format(l1regvf, l2regvf))
        regularizervf = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1regvf, scale_l2=l2regvf, scope="vf")
        all_trainable_weights_vf = vf_var_list
        regularization_penalty_vf = tf.contrib.layers.apply_regularization(regularizervf, all_trainable_weights_vf)
        vferr = vferr + regularization_penalty_vf

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])

    lossespi_inputs = [flat_tangent, ob, ac, atarg]
    lossesvf_inputs = [ob, ret]
    if todropoutpi:
        lossespi_inputs.append(dropoutpi_keep_prob)
    if todropoutvf:
        lossesvf_inputs.append(dropoutvf_keep_prob)
    if batchnormpi:
        lossespi_inputs.append(isbnpitrainmode)
    if batchnormvf:
        lossesvf_inputs.append(isbnvftrainmode)
    compute_losses = U.function(lossespi_inputs[1:], losses)
    compute_lossandgrad = U.function(lossespi_inputs[1:], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function(lossespi_inputs, fvp)
    compute_vflossandgrad = U.function(lossesvf_inputs, U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out

    U.initialize()
    if load_path is not None:
        pi.load(load_path)

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, nenvs, stochastic=True, 
            dropoutpi_keep_prob=(dropoutpi_keep_prob if todropoutpi else None),
            dropoutvf_keep_prob=(dropoutvf_keep_prob if todropoutvf else None),
            isbnpitrainmode=isbnpitrainmode,
            isbnvftrainmode=isbnvftrainmode)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    if toweightclippi:
         _wclip_ops_pi = []
         wclip_bounds_pi = [-weight_clip_range_pi, weight_clip_range_pi]
         for toclipvar in var_list:
             if 'logstd' in toclipvar.name:
                 continue
             _wclip_ops_pi.append(tf.assign(toclipvar, tf.clip_by_value(toclipvar, wclip_bounds_pi[0], wclip_bounds_pi[1])))
         _wclip_op_pi = tf.group(*_wclip_ops_pi)
    if toweightclipvf:
         _wclip_ops_vf = []
         wclip_bounds_vf = [-weight_clip_range_vf, weight_clip_range_vf]
         for toclipvar in vf_var_list:
             _wclip_ops_vf.append(tf.assign(toclipvar, tf.clip_by_value(toclipvar, wclip_bounds_vf[0], wclip_bounds_vf[1])))
         _wclip_op_vf = tf.group(*_wclip_ops_vf)

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam, nenvs)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        if isinstance(env.action_space, spaces.Discrete):
            seg["ob"] = seg["ob"].reshape(np.concatenate([[-1,seg["ob"].shape[1]],seg["ob"].shape[1:]]))
        args = seg["ob"], seg["ac"], atarg
        if todropoutpi:
            args = args + (network_kwargs['dropoutpi'],)
        if batchnormpi:
            args = args + (True,)

        fvpargs = [arr[::5] for arr in args if isinstance(arr, np.ndarray)]
        if todropoutpi:
            fvpargs.append(network_kwargs['dropoutpi'])
        if batchnormpi:
            fvpargs.append(True,)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
 
        if toweightclippi:
             U.get_session().run(_wclip_op_pi)

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        
        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=vf_batch_size):
                    vflossandgrad_inputs = (mbob, mbret)
                    if todropoutvf:
                        vflossandgrad_inputs += (network_kwargs["dropoutvf"],)
                    if batchnormvf:
                        vflossandgrad_inputs += (True,)

                    g = allmean(compute_vflossandgrad(*vflossandgrad_inputs))
                    if callable(vf_stepsize):
                        vfadam.update(g, vf_stepsize(timesteps_so_far/total_timesteps))
                    else:
                        vfadam.update(g, vf_stepsize)

        if toweightclipvf:
             U.get_session().run(_wclip_op_vf)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        meanret = [np.mean(arr) for arr in seg["ep_rets"]]
        meanret = sum(meanret)/len(meanret)
        totallen = [np.sum(arr) for arr in seg["ep_lens"]]
        totallen = sum(totallen)
        meanlen = [np.mean(arr) for arr in seg["ep_lens"]]
        meanlen = sum(meanlen)/len(meanlen)
        logger.record_tabular("EpLenMean", meanlen)
        logger.record_tabular("EpRewMean", meanret)
        totalepi = sum([len(arr) for arr in seg["ep_lens"]])
        logger.record_tabular("EpThisIter", totalepi)
        episodes_so_far += totalepi
        timesteps_so_far += totallen
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

