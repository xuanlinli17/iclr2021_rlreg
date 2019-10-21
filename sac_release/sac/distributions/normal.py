""" Multivariate normal distribution with mean and std deviation outputted by a neural net """

import tensorflow as tf
import numpy as np

from sac.misc.mlp import mlp

LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20


class Normal(object):
    def __init__(
            self,
            Dx,
            hidden_layers_sizes=(100, 100),
            reg=0.001,
            reparameterize=True,
            cond_t_lst=(),
            todropoutpi=False,
            dropoutpi=1.0,
            batchnormpi=False,
            isbnpitrainmode=None
    ):
        self._cond_t_lst = cond_t_lst
        self._reg = reg
        self._layer_sizes = list(hidden_layers_sizes) + [2 * Dx]
        print(self._layer_sizes)
        self._reparameterize = reparameterize
        self.todropoutpi = todropoutpi
        self.dropoutpi = dropoutpi
        self.batchnormpi = batchnormpi
        self.isbnpitrainmode = isbnpitrainmode
        self._Dx = Dx

        self._create_placeholders()
        self._create_graph()

    def _create_placeholders(self):
        self._N_pl = tf.placeholder(
            tf.int32,
            shape=(),
            name='N',
        )
        if self.todropoutpi:
            self.dropoutpi_placeholder = tf.placeholder(
                    dtype=tf.float32, 
                    shape=(),
                    name='dropoutpi_placeholder'
                    )


    def _create_graph(self):
        Dx = self._Dx

        if len(self._cond_t_lst) == 0:
            mu_and_logsig_t = tf.get_variable(
                'params', self._layer_sizes[-1],
                initializer=tf.random_normal_initializer(0, 0.1)
            )
        else:
            mu_and_logsig_t = mlp(
                inputs=self._cond_t_lst,
                layer_sizes=self._layer_sizes,
                output_nonlinearity=None,
                dropoutpi_placeholder=(self.dropoutpi_placeholder if self.todropoutpi else None),
                batchnorm=self.batchnormpi,
                isbnpitrainmode=self.isbnpitrainmode
            )  # ... x K*Dx*2+K

        self._mu_t = mu_and_logsig_t[..., :Dx]
        self._log_sig_t = tf.clip_by_value(mu_and_logsig_t[..., Dx:], LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)

        # Tensorflow's multivariate normal distribution supports reparameterization
        ds = tf.contrib.distributions
        dist = ds.MultivariateNormalDiag(loc=self._mu_t, scale_diag=tf.exp(self._log_sig_t))
        x_t = dist.sample()
        if not self._reparameterize:
            x_t = tf.stop_gradient(x_t)
        log_pi_t = dist.log_prob(x_t)

        self._dist = dist
        self._x_t = x_t
        self._log_pi_t = log_pi_t
        
        reg_loss_t = self._reg * 0.5 * tf.reduce_mean(self._log_sig_t ** 2)
        reg_loss_t += self._reg * 0.5 * tf.reduce_mean(self._mu_t ** 2)
        self._reg_loss_t = reg_loss_t



    @property
    def log_p_t(self):
        return self._log_pi_t

    @property
    def reg_loss_t(self):
        return self._reg_loss_t

    @property
    def x_t(self):
        return self._x_t

    @property
    def mu_t(self):
        return self._mu_t

    @property
    def log_sig_t(self):
        return self._log_sig_t
