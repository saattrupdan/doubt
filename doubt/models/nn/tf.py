''' Doubtful wrapper for Tensorflow models '''

import tensorflow as tf

class TFDoubt(tf.keras.Model):
    def __init__(self):
        super(TFDoubt, self).__init__()
        raise NotImplementedError
