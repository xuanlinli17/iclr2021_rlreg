import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, microbatch_size=None, 
                l1regpi, l2regpi, l1regvf, l2regvf, wclippi, wclipvf,
                todropoutpi, dropoutpi_keep_prob, dropoutpi_keep_prob_value,
                todropoutvf, dropoutvf_keep_prob, dropoutvf_keep_prob_value, 
                isbnpitrainmode, isbnvftrainmode):
        self.sess = sess = get_session()
        #REGULARIZATION
        self.toregularizepi = l1regpi > 0 or l2regpi > 0
        self.toregularizevf = l1regvf > 0 or l2regvf > 0
        self.todropoutpi = todropoutpi
        self.todropoutvf = todropoutvf
        self.dropoutpi_keep_prob = dropoutpi_keep_prob #TENSOR
        self.dropoutpi_keep_prob_value = dropoutpi_keep_prob_value
        self.dropoutvf_keep_prob = dropoutvf_keep_prob
        self.dropoutvf_keep_prob_value = dropoutvf_keep_prob_value
        self.isbnpitrainmode = isbnpitrainmode
        self.isbnvftrainmode = isbnvftrainmode
        self.toweightclippi = wclippi > 0
        self.toweightclipvf = wclipvf > 0

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)
            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef   
        if self.toregularizepi:
            print("regularizing policy network: L1 = {}, L2 = {}".format(l1regpi, l2regpi))
            regularizerpi = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1regpi, scale_l2=l2regpi, scope='ppo2_model/pi')
            all_trainable_weights_pi = tf.trainable_variables('ppo2_model/pi')
            regularization_penalty_pi = tf.contrib.layers.apply_regularization(regularizerpi, all_trainable_weights_pi)
            loss = loss + regularization_penalty_pi
        if self.toregularizevf:
            print("regularizing value network: L1 = {}, L2 = {}".format(l1regvf, l2regvf))
            regularizervf = tf.contrib.layers.l1_l2_regularizer(scale_l1=l1regvf, scale_l2=l2regvf, scope='ppo2_model/vf')
            all_trainable_weights_vf = tf.trainable_variables('ppo2_model/vf')
            regularization_penalty_vf = tf.contrib.layers.apply_regularization(regularizervf, all_trainable_weights_vf)
            loss = loss + regularization_penalty_vf

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        #self._update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(self._update_op):
        grads_and_var = self.trainer.compute_gradients(loss, params)

        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
       
        if self.toweightclippi:
             print("clipping policy network = {}".format(wclippi))
             policyparams = tf.trainable_variables('ppo2_model/pi') 
             self._wclip_ops_pi = []
             for toclipvar in policyparams:
                 if 'logstd' in toclipvar.name:
                     continue
                 self._wclip_ops_pi.append(tf.assign(toclipvar, tf.clip_by_value(toclipvar, -wclippi, wclippi)))
             self._wclip_op_pi = tf.group(*self._wclip_ops_pi)
        if self.toweightclipvf:
             print("clipping value network = {}".format(wclipvf))
             valueparams = tf.trainable_variables('ppo2_model/vf')
             self._wclip_ops_vf = []
             for toclipvar in valueparams:
                 self._wclip_ops_vf.append(tf.assign(toclipvar, tf.clip_by_value(toclipvar, -wclipvf, wclipvf)))
             self._wclip_op_vf = tf.group(*self._wclip_ops_vf)

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        if self.toregularizepi:
             self.loss_names.append('regularization_pi')
             self.stats_list.append(regularization_penalty_pi)
        if self.toregularizevf:
             self.loss_names.append('regularization_vf')
             self.stats_list.append(regularization_penalty_vf)

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if self.todropoutpi:
            td_map.update({self.dropoutpi_keep_prob: self.dropoutpi_keep_prob_value})
        if self.todropoutvf:
            td_map.update({self.dropoutvf_keep_prob: self.dropoutvf_keep_prob_value})
        if self.isbnpitrainmode is not None:
            td_map.update({self.isbnpitrainmode: True})
        if self.isbnvftrainmode is not None:
            td_map.update({self.isbnvftrainmode: True})
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks
       
        vartorun = self.stats_list + [self._train_op]
        vartorun_len = len(vartorun)
        
        if self.toweightclippi:
            vartorun = vartorun + [self._wclip_op_pi]
        if self.toweightclipvf:
            vartorun = vartorun + [self._wclip_op_vf]
        return self.sess.run(vartorun, td_map)[:vartorun_len-1]
        
