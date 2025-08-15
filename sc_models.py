import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Layer, Dense, Concatenate, Lambda, LSTM, LSTMCell, RNN, Dropout, \
    LayerNormalization, Activation, Lambda
tf.keras.backend.set_floatx('float32')
dtype = tf.keras.backend.floatx()


def hard_dec(x):
    return tf.where(tf.greater(x, 0), 1.0, 0.0)


class CheckNodeVanilla(Model):
    def __init__(self, clip=10000.0, name='checknode'):
        super(CheckNodeVanilla, self).__init__(name=name)
        self.clip = clip

    def call(self, inputs, **kwargs):
        e1, e2 = inputs
        return tf.clip_by_value(-2 * tf.math.atanh(tf.math.tanh(e1 / 2) * tf.math.tanh(e2 / 2)), -self.clip, self.clip)


class CheckNodeMinSum(Model):
    def __init__(self, name='checknode_minsum'):
        super(CheckNodeMinSum, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        e1, e2 = inputs
        return tf.math.sign(e1)*tf.math.sign(e2)*tf.minimum(tf.abs(e1), tf.abs(e2))


class BitNodeVanilla(Model):
    def __init__(self, name='bitnode'):
        super(BitNodeVanilla, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        e1, e2, uhat = inputs
        return e2 + (1. - 2. * tf.cast(uhat, dtype)) * e1


class CheckNodeTrellis(Model):
    def __init__(self, batch_dims=2, state_size=2, name='checknode'):
        super(CheckNodeTrellis, self).__init__(name=name)
        self.batch_dims = batch_dims
        self.state_size = state_size

    def call(self, inputs, **kwargs):
        e1, e2 = inputs
        s0, u1, s2 = tf.meshgrid(tf.range(tf.shape(e1)[-2]), [0, 1], tf.range(tf.shape(e1)[-1]))
        arg = tf.ones_like(u1)
        res_ = list()
        # tf.autograph.experimental.set_loop_options(
        #     shape_invariants=[(res_, tf.shape(e1))]
        # )
        repmat = tf.concat((tf.shape(e1)[:self.batch_dims], [1, 1, 1, 1]), axis=0)
        for u2 in range(2):
            for s1 in range(self.state_size):
                arg1 = tf.stack([tf.math.floormod(u1 + u2, 2), s0, s1 * arg], axis=-1)
                for _ in range(self.batch_dims):
                    arg1 = tf.expand_dims(arg1, 0)
                indices1 = tf.tile(arg1, repmat)
                arg2 = tf.stack([u2 * arg, s1 * arg, s2], axis=-1)
                for _ in range(self.batch_dims):
                    arg2 = tf.expand_dims(arg2, 0)
                indices2 = tf.tile(arg2, repmat)
                res_.append(tf.gather_nd(e1, indices1, batch_dims=self.batch_dims) +
                            tf.gather_nd(e2, indices2, batch_dims=self.batch_dims))
        res_ = tf.stack(res_, axis=-1)
        # res = tf.reduce_logsumexp(tf.math.abs(res_), axis=-1) * tf.reduce_prod(tf.math.sign(res_), axis=-1)
        res = tf.reduce_logsumexp(res_, axis=-1)

        # res = res_ / tf.reduce_mean(tf.reduce_sum(res_, axis=(2, 3, 4)))
        return res


class BitNodeTrellis(Model):
    def __init__(self, batch_dims=2, state_size=2, name='bitnode'):
        super(BitNodeTrellis, self).__init__(name=name)
        self.batch_dims = batch_dims
        self.state_size = state_size

    def call(self, inputs, **kwargs):
        e1, e2, uhat = inputs

        s0, u2, s2 = tf.meshgrid(tf.range(tf.shape(e1)[-2]), [0, 1], tf.range(tf.shape(e1)[-1]))
        uhat_ = tf.cast(uhat, tf.int32)
        arg0 = tf.expand_dims(tf.expand_dims(uhat_, -1), -1)
        repmat = tf.concat((tf.ones([self.batch_dims], tf.int32), tf.shape(e1)[self.batch_dims:]), axis=0)
        uhat_t = tf.tile(arg0, repmat)
        u2_tiled = u2
        for i in range(self.batch_dims):
            u2_tiled = tf.expand_dims(u2_tiled, 0)
        repmat = tf.concat((tf.shape(e1)[:self.batch_dims], [1, 1, 1]), axis=0)
        u2_t = tf.tile(u2_tiled, repmat)
        u_xor = tf.math.floormod(u2_t + uhat_t, 2)

        arg = tf.ones_like(u2)
        res_ = list()
        repmat = tf.concat((tf.shape(e1)[:self.batch_dims], [1, 1, 1, 1]), axis=0)

        for s1 in range(self.state_size):
            arg1 = tf.stack([s0, s1 * arg], axis=-1)
            for i in range(self.batch_dims):
                arg1 = tf.expand_dims(arg1, 0)
            indices1 = tf.tile(arg1, repmat)
            indices1 = tf.concat([tf.expand_dims(u_xor, -1), indices1], axis=-1)

            arg2 = tf.stack([u2 * arg, s1 * arg, s2], axis=-1)
            for i in range(self.batch_dims):
                arg2 = tf.expand_dims(arg2, 0)
            indices2 = tf.tile(arg2, repmat)

            res_.append(tf.gather_nd(e1, indices1, batch_dims=self.batch_dims) +
                        tf.gather_nd(e2, indices2, batch_dims=self.batch_dims))

        res_ = tf.stack(res_, axis=-1)
        res = tf.reduce_logsumexp(res_, axis=-1)
        return res


class Embedding2LLRTrellis(Model):
    def __init__(self, batch_dims=2, name='emb2prob_trellis'):
        super(Embedding2LLRTrellis, self).__init__(name=name)
        self.batch_dims = batch_dims

    def call(self, inputs, training=None, **kwargs):
        e = inputs

        e1 = tf.squeeze(tf.gather(e, indices=[1], axis=self.batch_dims), axis=self.batch_dims)
        e0 = tf.squeeze(tf.gather(e, indices=[0], axis=self.batch_dims), axis=self.batch_dims)
        p1 = tf.reduce_logsumexp(e1, axis=(-1, -2))
        p0 = tf.reduce_logsumexp(e0, axis=(-1, -2))
        # tf.einsum(f'{self.ein_str}lm->{self.ein_str}', e1)
        # p0 = tf.einsum(f'{self.ein_str}lm->{self.ein_str}', e0)
        # e = tf.sigmoid(tf.math.log(tf.where(p1 > p0, 2., 0.5)))
        e = tf.cast(p1-p0, tf.float32)
        return tf.expand_dims(e, -1)


class EmbeddingX(Model):
    def __init__(self, logits, name='embedding_x'):
        """

        Returns:
            object:
        """
        super(EmbeddingX, self).__init__(name=name)
        self.input_logits = self.add_weight(name="logits", shape=logits.shape.as_list(), dtype=dtype, trainable=True,
                                            initializer=tf.keras.initializers.constant(logits))
        self.logits_shape = (1,) if len(tf.shape(self.input_logits)) == 0 else tf.shape(self.input_logits)

    def call(self, inputs, training=None, **kwargs):
        e = tf.broadcast_to(self.input_logits,
                            shape=tf.concat([inputs, self.logits_shape], axis=0))
        return e


class EmbeddingY(Model):
    def __init__(self, hidden_size, embedding_size, activation='elu', use_bias=True,
                 layer_normalization=False, name='emb_y'):
        super(EmbeddingY, self).__init__(name=name)

        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer1"),
                        Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer2")]

        if layer_normalization:
            self._layers.insert(0, tf.keras.layers.LayerNormalization())

    def call(self, inputs, training=None, **kwargs):
        e = inputs
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        return e


class CheckNodeNNEmb(Model):
    def __init__(self, hidden_size, embedding_size, layers_per_op, activation='elu',
                 use_bias=True, name='checknode_nnops'):
        super(CheckNodeNNEmb, self).__init__(name=name)

        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}")
                        for i in range(layers_per_op)] + \
                       [Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}")]

    def call(self, inputs, training=None, **kwargs):
        e1, e2 = inputs
        e = tf.concat([e1, e2], axis=-1)
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        return e


class BitNodeNNEmb(Model):
    def __init__(self, hidden_size, embedding_size, layers_per_op, activation='elu',
                 use_bias=True, name='bitnode_nnops'):
        super(BitNodeNNEmb, self).__init__(name=name)
        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}")
                        for i in range(layers_per_op)] + \
                       [Dense(embedding_size, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}")]

    def call(self, inputs, training=None, **kwargs):
        e1, e2, u = inputs
        e = tf.concat([e1, e2, tf.cast(u, dtype)], axis=-1)
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        return e


class Embedding2LLR(Model):
    def __init__(self, hidden_size, layers_per_op, activation='elu', use_bias=True, name='emb2llr_nnops'):
        super(Embedding2LLR, self).__init__(name=name)

        self._layers = [Dense(hidden_size, activation=activation, use_bias=use_bias, name=f"{name}-layer{i}")
                        for i in range(layers_per_op)] + \
                       [Dense(1, activation=None, use_bias=use_bias, name=f"{name}-layer{layers_per_op}")]

    def call(self, inputs, training=None, **kwargs):
        e = inputs
        for layer in self._layers:
            e = layer.__call__(e, training=training)
        return e
