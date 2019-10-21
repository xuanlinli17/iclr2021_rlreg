import tensorflow as tf
from rllab import config


def get_default_session():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.09)
    return tf.get_default_session() or create_session(config=tf.ConfigProto(gpu_options=gpu_options))
    #return create_session(config=tf.ConfigProto(gpu_options=gpu_options))

def create_session(**kwargs):
    """ Create new tensorflow session with given configuration. """
    if "config" not in kwargs:
        kwargs["config"] = get_configuration()
    return tf.InteractiveSession(**kwargs)


def get_configuration():
    """ Returns personal tensorflow configuration. """
    if config.USE_GPU:
        raise NotImplementedError

    config_args = dict()
    return tf.ConfigProto(**config_args)
