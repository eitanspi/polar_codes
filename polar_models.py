import re
import os
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, SGD
from sionna.fec.crc import CRCEncoder, CRCDecoder
from models.sc_models import CheckNodeVanilla, BitNodeVanilla, CheckNodeTrellis, BitNodeTrellis, \
    BitNodeNNEmb, CheckNodeNNEmb, Embedding2LLR, Embedding2LLRTrellis, EmbeddingY, EmbeddingX, hard_dec
from models.layers import SplitEvenOdd, Interleave, F2
from models.input_models import BinaryRNN
import wandb
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
tf.keras.backend.set_floatx('float32')
dtype = tf.keras.backend.floatx()


class SCDecoder(Model):

    def __init__(self, channel, batch=100, *args, **kwargs):
        Model.__init__(self)

        self.channel = channel
        self.batch = batch

        self.llr_enc_shape = self.llr_dec_shape = (1,)
        self.input_logits = tf.constant(0.0, dtype=dtype)
        self.Ex = self.Ex_enc = EmbeddingX(self.input_logits)

        self.Ey = self.channel.llr
        self.checknode_enc = self.checknode = CheckNodeVanilla()
        self.bitnode_enc = self.bitnode = BitNodeVanilla()
        self.emb2llr_enc = self.emb2llr = Activation(tf.identity)

        self.llr2prob = Activation(tf.math.sigmoid)
        self.split_even_odd = SplitEvenOdd(axis=1)
        self.f2 = F2()
        self.interleave = Interleave()
        self.hard_decision = Activation(hard_dec)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                          reduction=tf.keras.losses.Reduction.NONE)

    @tf.function
    def encode(self, e, f, N, r, sample=True):
        """ sample=True indicates sampling bits at the recursion leaves.
            sample=False indicates taking the argmax at the recursion leaves.
        """
        if N == 1:
            llr_u = self.emb2llr_enc(e)
            if sample:
                pu = tf.math.sigmoid(llr_u)
                sampled_u = tf.where(tf.greater_equal(r, pu), 0.0, 1.0)
                x = tf.where(tf.equal(f, 0.5), sampled_u, f)
            else:
                x = tf.where(tf.equal(f, 0.5),
                             self.hard_decision(llr_u),
                             f)
            return x, tf.identity(x), llr_u

        e_odd, e_even = self.split_even_odd.call(e)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=1)
        r_left, r_right = r_halves

        # Compute soft mapping back one stage
        u1est = self.checknode_enc.call((e_odd, e_even))

        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, llr_u_left = self.encode(u1est, f_left, N // 2, r_left, sample=sample)

        # Using u1est and x1hard, we can estimate u2
        u2est = self.bitnode_enc.call((e_odd, e_even, u1hardprev))

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, llr_u_right = self.encode(u2est, f_right, N // 2, r_right, sample=sample)

        u = tf.concat([uhat1, uhat2], axis=1)
        llr_u = tf.concat([llr_u_left, llr_u_right], axis=1)
        x = self.f2.call((u1hardprev, u2hardprev))
        x = self.interleave.call(x)
        return u, x, llr_u

    @tf.function
    def decode(self, ex, ey, f, N, r, sample=True):
        if N == 1:
            llr_u = self.emb2llr(ex)
            if sample:
                pu = tf.math.sigmoid(llr_u)
                frozen = tf.where(tf.greater_equal(r, pu), 0.0, 1.0)
            else:
                frozen = self.hard_decision(llr_u)

            llr_uy = self.emb2llr(ey)
            is_frozen = tf.logical_or(tf.equal(f, 0.0), tf.equal(f, 1.0))
            x = tf.where(is_frozen, frozen, self.hard_decision(llr_uy))
            return x, tf.identity(x), llr_u, llr_uy

        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=1)
        r_left, r_right = r_halves

        ey_odd, ey_even = self.split_even_odd.call(ey)
        ex_odd, ex_even = self.split_even_odd.call(ex)
        e_odd, e_even = tf.concat([ex_odd, ey_odd], axis=0), tf.concat([ex_even, ey_even], axis=0)

        # Compute soft mapping back one stage
        e1est = self.checknode.call((e_odd, e_even))
        ex1est, ey1est = tf.split(e1est, num_or_size_splits=2, axis=0)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, llr_u_left, llr_uy_left = self.decode(ex1est, ey1est, f_left, N // 2, r_left,
                                                                  sample=sample)

        # Using u1est and x1hard, we can estimate u2
        u1_hardprev_dup = tf.tile(u1hardprev, [2, 1, 1])
        e2est = self.bitnode.call((e_odd, e_even, u1_hardprev_dup))
        ex2est, ey2est = tf.split(e2est, num_or_size_splits=2, axis=0)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, llr_u_right, llr_uy_right = self.decode(ex2est, ey2est, f_right, N // 2, r_right,
                                                                    sample=sample)

        u = tf.concat([uhat1, uhat2], axis=1)
        llr_u = tf.concat([llr_u_left, llr_u_right], axis=1)
        llr_uy = tf.concat([llr_uy_left, llr_uy_right], axis=1)
        x = self.f2.call((u1hardprev, u2hardprev))
        x = self.interleave.call(x)
        return u, x, llr_u, llr_uy

    @tf.function
    def transform(self, u, N):
        if N == 1:
            return u
        else:
            # R_N maps odd/even indices (i.e., u1u2/u2) to first/second half
            # Compute odd/even outputs of (I_{N/2} \otimes G_2) transform
            u_odd, u_even = self.split_even_odd.call(u)
            x_left = self.transform(tf.math.floormod(u_odd + u_even, 2), N // 2)
            x_right = self.transform(u_even, N // 2)
            return tf.concat([x_left, x_right], axis=1)

    @tf.function
    def fast_ce(self, emb, x):
        depth = emb.shape[1]
        num_of_splits = 1

        loss_array = tf.TensorArray(dtype=tf.float32, size=int(np.log2(depth) + 1))
        errors_array = tf.TensorArray(dtype=tf.float32, size=int(np.log2(depth) + 1))

        pred = self.emb2llr.call(emb)
        loss = self.loss_fn(x, pred)
        loss_array = loss_array.write(0, loss)
        errors = tf.squeeze(tf.cast(tf.not_equal(x, self.hard_decision(pred)), tf.float32), axis=2)
        errors_array = errors_array.write(0, errors)
        V = list([x])
        E = list([emb])
        # iterate over decoding stage
        while depth > 1:
            V_1 = list([])
            V_2 = list([])
            E_1 = list([])
            E_2 = list([])
            # split into even and odd indices with respect to the depth
            for v, e in zip(V, E):
                # compute bits amd embeddings in next layer
                v_odd, v_even = self.split_even_odd.call(v)
                V_1.append(v_odd)
                V_2.append(v_even)
                e_odd, e_even = self.split_even_odd.call(e)
                E_1.append(e_odd)
                E_2.append(e_even)

            # compute all the bits in the next stage
            V_odd = tf.concat(V_1, axis=1)
            V_even = tf.concat(V_2, axis=1)
            v_xor = tf.math.floormod(V_odd + V_even, 2)
            V_xor = tf.split(v_xor, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            V_identity = tf.split(V_even, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            v = tf.concat([elem for pair in zip(V_xor, V_identity) for elem in pair], axis=1)
            V_ = tf.split(v, num_or_size_splits=2 ** num_of_splits, axis=1)

            # compute all the embeddings in the next stage
            E_odd = tf.concat(E_1, axis=1)
            E_even = tf.concat(E_2, axis=1)
            V_left = tf.concat(V_[::2], axis=1)
            e1_left = self.checknode.call((E_odd, E_even))
            e1_right = self.bitnode.call((E_odd, E_even, V_left))
            E1_left = tf.split(e1_left, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            E1_right = tf.split(e1_right, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            e_lr = tf.concat([elem for pair in zip(E1_left, E1_right) for elem in pair], axis=1)
            E_ = tf.split(e_lr, num_or_size_splits=2 ** num_of_splits, axis=1)

            # on the last depth compute the CE of the synthetic channels
            e = tf.concat(E_, axis=1)
            v = tf.concat(V_, axis=1)
            pred = self.emb2llr.call(e)
            loss = self.loss_fn(v, pred)
            loss_array = loss_array.write(num_of_splits, loss)

            errors = tf.squeeze(tf.cast(tf.not_equal(v, self.hard_decision(pred)), tf.float32), axis=2)
            errors_array = errors_array.write(num_of_splits, errors)

            V = V_
            E = E_

            depth //= 2
            num_of_splits += 1

        stacked_loss = loss_array.stack()
        stacked_loss = tf.transpose(stacked_loss, perm=[1, 0, 2])
        stacked_errors = errors_array.stack()
        stacked_errors = tf.transpose(stacked_errors, perm=[1, 0, 2])
        return stacked_loss, stacked_errors

    @tf.function
    def forward_design(self, batch, N):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        ex_enc = self.Ex_enc.call(batch_N_shape)

        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)

        # create frozen bits for encoding. encoded bits need to be 0.5.
        f_enc = 0.5 * tf.ones(shape=(batch, N, 1))
        u, x, llr_u1 = self.encode(ex_enc, f_enc, N, r, sample=True)
        llr_u = tf.squeeze(tf.where(tf.equal(u, 1.0), llr_u1, -llr_u1), axis=2)
        h_u = tf.reduce_sum(-tf.math.log(tf.math.sigmoid(llr_u)), axis=0)

        y = self.channel.sample_channel_outputs(x)
        ex = self.Ex.call(batch_N_shape)
        ey = self.Ey(y)
        loss_array, errors_array = self.fast_ce(ex + ey, x)
        h_uy = tf.reduce_sum(loss_array[:, -1, :], axis=0)
        errors = tf.reduce_sum(errors_array[:, -1, :], axis=0)
        return h_u, h_uy, errors

    @tf.function
    def forward_eval(self, batch, N, info_indices, frozen_indices, A, Ac):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        # generate the information bits
        bits = tf.cast(tf.random.uniform((batch, N), minval=0, maxval=2, dtype=tf.int32), dtype)

        # create frozen bits for encoding. encoded bits need to be 0.5. info bit 0/1
        updates = 0.5 * tf.ones([batch * tf.shape(Ac)[0]], dtype=dtype)
        f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(bits, frozen_indices, updates), axis=2)

        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)

        # encode the bits into x^N and u^N
        ex_enc = self.Ex_enc.call(batch_N_shape)
        u, x, llr_u1_enc = self.encode(ex_enc, f_enc, N, r, sample=True)
        y = self.channel.sample_channel_outputs(x)

        # create frozen bits for decoding. decoded bits need to be 0.5.
        # frozen bits are 0 decoded using argmax like in the encoder
        tensor = tf.zeros(shape=(batch, N))
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)

        # decode and compute the errors
        ey = self.Ey(y)
        ex = self.Ex.call(batch_N_shape)

        uhat, xhat, llr_u1, llr_uy1 = self.decode(ex, ex+ey, f_dec, f_dec.shape[1], r, sample=True)
        errors = tf.squeeze(tf.cast(tf.where(tf.equal(uhat, u), 0, 1), dtype), axis=-1)
        info_bit_errors = tf.gather(params=errors,
                                    indices=A,
                                    axis=1)
        return tf.reduce_mean(info_bit_errors, axis=1)

    def polar_code_design(self, n, batch, mc, tol=100):
        biterrd = tf.zeros([2 ** n], dtype=dtype)
        Hu = tf.zeros([2 ** n], dtype=dtype)
        Huy = tf.zeros([2 ** n], dtype=dtype)

        count = 0
        stop_err = max((2**n)//4, 3)

        def stop_criteria(arr):
            return np.sum(arr < tol)

        t = time()
        while stop_criteria(biterrd) > stop_err and count < mc:
            h_u, h_uy, errors = self.forward_design(batch, 2 ** n)
            Hu += h_u
            Huy += h_uy
            biterrd += errors
            count += batch

            if time()-t > 60:
                print(f'iter: {count/mc*100 :5.3f}% | bits w/o {tol} errors {stop_criteria(biterrd)} > {stop_err}')
                t = time()

        print(f'iter: {count/mc*100}% | bits w/o {tol} errors {stop_criteria(biterrd)} > {stop_err}')

        biterrd /= count
        Hu /= count
        Huy /= count
        print(f'conditional entropies of effective bit channels:\n'
              f'Hu: {tf.reduce_mean(Hu).numpy()/np.log(2) : 5.4f} Huy: {tf.reduce_mean(Huy).numpy()/np.log(2) : 5.4f} '
              f'MI: {tf.reduce_mean(Hu - Huy).numpy()/np.log(2) : 5.4f}')
        mi = np.squeeze(Hu - Huy)
        sorted_arg_errors = np.argsort(-mi)

        return Hu/np.log(2), Huy/np.log(2), sorted_arg_errors

    def polar_code_err_prob(self, n, mc_err, batch, sorted_bit_channels, k):
        A, Ac = self.choose_information_and_frozen_sets(sorted_bit_channels, k)

        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(A, tf.int32))
        info_indices = tf.stack([tf.reshape(tf.transpose(X, perm=[1, 0]), -1),
                                 tf.reshape(tf.transpose(Y, perm=[1, 0]), -1)], axis=1)
        X, Y = tf.meshgrid(tf.range(batch, dtype=tf.int32), tf.cast(Ac, tf.int32))
        frozen_indices = tf.stack([tf.reshape(X, -1), tf.reshape(Y, -1)], axis=1)

        mc_err = (mc_err // batch + 1) * batch
        err = np.zeros(shape=0)
        t = time()
        for i in range(0, mc_err, batch):
            bit_errors = self.forward_eval(batch, 2 ** n, info_indices, frozen_indices, A, Ac)
            err = np.concatenate((err, bit_errors))
            if time()-t > 60:
                ber = np.mean(err)
                fer = np.mean(err > 0)
                print(f'iter: {i/mc_err*100 :5.3f}% | ber: {ber : 5.3e} fer {fer : 5.3e}')
                t = time()
        return err

    def eval(self, Ns, mc_length=100000, code_rate=0.25, batch=100, tol=100, load_nsc_path=None, design_path=None, design_load=False):

        if load_nsc_path is not None:
            self.load_model(load_nsc_path)
            decoder_name = load_nsc_path
        else:
            decoder_name = 'sc'

        bers, fers = list(), list()
        for n in Ns:
            log_dict = {}
            print(n)
            t = time()
            if design_path is None:
                try:
                    if design_load:
                        design_name = f"{self.construction_name(n, decoder_name)}:latest"
                        sorted_bit_channels = self.load_design(design_name)
                        print(f"Design loaded: {design_name}")
                    else:
                        raise Exception("design_load flags is False")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    Hu, Huy, sorted_bit_channels = self.polar_code_design(n, batch, 10 ** 7, tol=tol)
                    self.save_design(n, sorted_bit_channels, decoder_name)
                    log_dict.update(self.design2dict(n, Hu, Huy))
            else:
                sorted_bit_channels = self.load_design(design_path)

            design_time = time() - t
            k = int(code_rate * (2 ** n))

            t = time()
            err = self.polar_code_err_prob(n, mc_length, batch, sorted_bit_channels, k)
            mc_time = time() - t
            ber = np.mean(err)
            fer = np.mean(err > 0)
            bers.append(ber)
            fers.append(fer)
            print(f"n: {n: 2d} design time: {design_time: 4.1f} "
                  f"code rate: {code_rate: 5.4f} #of mc-blocks: {mc_length} mc time: {mc_time: 4.1f} "
                  f"ber: {ber: 4.3e} fer: {fer: 4.3e}")
            log_dict.update({"n": n,
                             "ber": ber,
                             "fer": fer,
                             "code_rate": code_rate})
            wandb.log(log_dict)

        if len(Ns) != 1:
            x_values = np.array(Ns)
            y_values = np.array(bers)
            z_values = np.array(fers)
            data = [[x, y, z] for (x, y, z) in zip(x_values, y_values, z_values)]
            table = wandb.Table(data=data, columns=["n", "ber", "fer"])
            wandb.log({f"ber": table})

    def choose_information_and_frozen_sets(self, sorted_bit_channels, k):
        A = sorted_bit_channels[:k]
        A = sorted(A)
        Ac = sorted_bit_channels[k:]
        Ac = sorted(Ac)
        return A, Ac

    def set_input_logits(self, logits):
        self.Ex.input_logits.assign(logits)
        self.Ex_enc.input_logits.assign(logits)

    def construction_name(self, n, decoder_name):
        if len(decoder_name.split("/")) > 1:
            decoder_name = decoder_name.split("/")[-1]
        save_name = f'dec-{decoder_name}__channel-{self.channel.save_name}__n-{n}'.replace(':', "-")
        return save_name

    def save_design(self, n, sorted_bit_channels, decoder_name):
        try:
            save_name = self.construction_name(n, decoder_name)
            save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/design.npy"
            # Extract the directory part of the path
            directory = os.path.dirname(save_tmp_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(save_tmp_path, sorted_bit_channels)
            weights_artifact = wandb.Artifact(save_name, type='design')

            # Add the model weights file to the artifact
            weights_artifact.add_file(save_tmp_path)

            # Log the artifact to W&B
            artifact_log = wandb.log_artifact(weights_artifact)
            artifact_log.wait()
            # print(f"model is saved as {save_name}")
            artifact = wandb.run.use_artifact(f"{save_name}:latest")
            print(f"Design: {artifact.name}")
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def load_design(load_path):
        artifact = wandb.use_artifact(load_path, type='design')
        artifact_dir = artifact.download()

        array_data = np.load(f'{artifact_dir}/design.npy')
        print(f"design is loaded from {load_path}")
        return array_data

    @staticmethod
    def log2wandb(n, ber, fer, Hu, Huy, code_rate):
        log_dict = {"n": n,
                    "ber": ber,
                    "fer": fer,
                    "code_rate": code_rate}
        Hu = np.squeeze(Hu)
        Huy = np.squeeze(Huy)
        mi = np.squeeze(Hu - Huy)

        x_values = np.array(range(len(Hu))).astype(int)
        y_values = np.array(np.sort(Hu))
        data_Hu = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Hu = wandb.Table(data=data_Hu, columns=["channel_idx", "Hu"])
        y_values = np.array(np.sort(Huy))
        data_Huy = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Huy = wandb.Table(data=data_Huy, columns=["channel_idx", "Huy"])
        y_values = np.array(np.sort(mi))
        data_mi = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_mi = wandb.Table(data=data_mi, columns=["channel_idx", "mi"])
        log_dict.update({f"Hu-n-{n}": wandb.plot.scatter(table_Hu, "channel_idx", "Hu", title=f"Hu-n-{n}"),
                         f"Huy-n-{n}": wandb.plot.scatter(table_Huy, "channel_idx", "Huy", title=f"Huy-n-{n}"),
                         f"mi-n-{n}": wandb.plot.scatter(table_mi, "channel_idx", "mi", title=f"mi-n-{n}")})
        wandb.log(log_dict)

    @staticmethod
    def design2dict(n, Hu, Huy):
        Hu = np.squeeze(Hu)
        Huy = np.squeeze(Huy)
        mi = np.squeeze(Hu - Huy)

        x_values = np.array(range(len(Hu))).astype(int)
        y_values = np.array(np.sort(Hu))
        data_Hu = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Hu = wandb.Table(data=data_Hu, columns=["channel_idx", "Hu"])
        y_values = np.array(np.sort(Huy))
        data_Huy = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_Huy = wandb.Table(data=data_Huy, columns=["channel_idx", "Huy"])
        y_values = np.array(np.sort(mi))
        data_mi = [[x, y] for (x, y) in zip(x_values, y_values)]
        table_mi = wandb.Table(data=data_mi, columns=["channel_idx", "mi"])
        log_dict = {f"Hu-n-{n}": wandb.plot.scatter(table_Hu, "channel_idx", "Hu", title=f"Hu-n-{n}"),
                    f"Huy-n-{n}": wandb.plot.scatter(table_Huy, "channel_idx", "Huy", title=f"Huy-n-{n}"),
                    f"mi-n-{n}": wandb.plot.scatter(table_mi, "channel_idx", "mi", title=f"mi-n-{n}")}
        return log_dict


class SCTrellisDecoder(SCDecoder):

    def __init__(self, channel, batch=100, *args, **kwargs):
        SCDecoder.__init__(self, channel, batch, *args, **kwargs)

        self.llr_enc_shape = self.llr_dec_shape = (self.channel.cardinality_x,
                                                   self.channel.cardinality_s,
                                                   self.channel.cardinality_s)

        unnormalized_logits = tf.zeros(shape=(self.channel.cardinality_x,
                                              self.channel.cardinality_s,
                                              self.channel.cardinality_s))
        logits = unnormalized_logits - tf.math.reduce_logsumexp(unnormalized_logits, axis=(0, 2), keepdims=True)
        self.input_logits = tf.constant(logits, dtype=dtype)
        self.Ex = self.Ex_enc = EmbeddingX(self.input_logits)

        self.checknode_enc = self.checknode = CheckNodeTrellis(state_size=self.channel.cardinality_s)
        self.bitnode_enc = self.bitnode = BitNodeTrellis(state_size=self.channel.cardinality_s)
        self.emb2llr_enc = self.emb2llr = Embedding2LLRTrellis()


class SCNeuralDecoder(SCDecoder):

    def __init__(self, channel, embedding_size, hidden_size, layers_per_op,
                 activation='elu', batch=100, *args, **kwargs):
        SCDecoder.__init__(self, channel, batch, *args, **kwargs)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers_per_op = layers_per_op
        self.activation = activation

        self.llr_dec_shape = (embedding_size,)
        self.Ey = Sequential([Dense(embedding_size, use_bias=True, activation=None)])
        self.Ex_enc = self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size,)))
        self.checknode = self.checknode_enc = CheckNodeNNEmb(hidden_size, embedding_size, layers_per_op,
                                                             use_bias=True, activation=activation)
        self.bitnode = self.bitnode_enc = BitNodeNNEmb(hidden_size, embedding_size, layers_per_op,
                                                       use_bias=True, activation=activation)
        self.emb2llr = self.emb2llr_enc = Embedding2LLR(hidden_size, layers_per_op,
                                                        use_bias=True, activation=activation)

    def load_model(self, load_path):
        artifact = wandb.use_artifact(load_path, type='model_weights')
        artifact_dir = artifact.download()

        shape = (self.batch, 4)
        u = tf.zeros(shape=(self.batch, 4) + (1,))
        y = tf.zeros(shape=(self.batch, 4) + (self.channel.cardinality_y,))
        e_dec = self.Ex.call(shape)
        self.Ey(y)
        self.checknode.call((e_dec, e_dec))
        self.bitnode.call((e_dec, e_dec, u))
        self.emb2llr.call(e_dec)
        self.checknode.built = True
        self.bitnode.built = True
        self.emb2llr.built = True
        self.Ex.built = True
        self.Ey.built = True
        self.checknode.load_weights(f"{artifact_dir}/checknode.weights.h5")
        self.bitnode.load_weights(f"{artifact_dir}/bitnode.weights.h5")
        self.emb2llr.load_weights(f"{artifact_dir}/emb2llr.weights.h5")
        self.Ex.load_weights(f"{artifact_dir}/ex.weights.h5")
        self.Ey.load_weights(f"{artifact_dir}/ey.weights.h5")

        self.Ex_enc.call(shape)
        self.Ex_enc.built = True
        self.Ex_enc.load_weights(f"{artifact_dir}/ex.weights.h5", by_name=True, skip_mismatch=True)
        print(f"model is loaded from {load_path}")


class NeuralPolarDecoder(SCDecoder):
    def __init__(self, channel, embedding_size, hidden_size, layers_per_op,
                 activation='elu', batch=100, lr=0.001, optimizer='sgd',
                 input_distribution='sc', input_state_size=2):
        SCDecoder.__init__(self, channel, batch=batch)

        self.channel = channel
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers_per_op = layers_per_op
        self.activation = activation
        self.lr = lr

        self.input_distribution = input_distribution
        if input_distribution == 'sc':
            self.llr_enc_shape = ()
            logits = tf.zeros(shape=self.llr_enc_shape)
            self.input_logits = tf.constant(logits, dtype=dtype)
            self.Ex_enc = EmbeddingX(self.input_logits)
            self.checknode_enc = CheckNodeVanilla()
            self.bitnode_enc = BitNodeVanilla()
            self.emb2llr_enc = Activation(tf.identity)
            self.input_distribution_name = f"{self.input_distribution}"
            print("Input distribution is implemented via a SC encoder")
        elif input_distribution == 'sct':
            self.llr_enc_shape = (2, input_state_size, input_state_size)
            unnormalized_logits = tf.zeros(shape=self.llr_enc_shape)
            logits = unnormalized_logits - tf.math.reduce_logsumexp(unnormalized_logits, axis=(0, 2), keepdims=True)
            self.input_logits = tf.constant(logits, dtype=dtype)
            self.Ex_enc = EmbeddingX(self.input_logits)
            self.checknode_enc = CheckNodeTrellis(state_size=self.channel.cardinality_s)
            self.bitnode_enc = BitNodeTrellis(state_size=self.channel.cardinality_s)
            self.emb2llr_enc = Embedding2LLRTrellis()
            self.input_distribution_name = f"{self.input_distribution}-state-size-{input_state_size}"
            print("Input distribution is implemented via a SCT encoder")
        elif input_distribution == 'rnn':
            self.input_model = BinaryRNN(units=embedding_size)
            self.input_distribution_name = f"{self.input_distribution}-hidden-{embedding_size}"
            print("Input distribution is implemented via an RNN encoder")
        else:
            raise ValueError(f'input_distribution received invalid value: {input_distribution}')

        self.llr_dec_shape = (embedding_size,)
        self.Ey = Sequential([Dense(embedding_size, use_bias=True, activation=None)])
        self.Ex = EmbeddingX(tf.zeros(shape=(embedding_size,)))
        self.checknode = CheckNodeNNEmb(hidden_size, embedding_size, layers_per_op,
                                        use_bias=True, activation=activation)
        self.bitnode = BitNodeNNEmb(hidden_size, embedding_size, layers_per_op,
                                    use_bias=True, activation=activation)
        self.emb2llr = Embedding2LLR(hidden_size, layers_per_op,
                                     use_bias=True, activation=activation)

        self.split_even_odd = SplitEvenOdd(axis=1)
        self.f2 = F2()
        self.interleave = Interleave(axis=1)

        self.optimizer_name = optimizer
        if optimizer == "sgd":
            self.estimation_optimizer = SGD(learning_rate=lr, clipnorm=1.0)
            self.improvement_optimizer = SGD(learning_rate=lr, clipnorm=1.0)
        elif optimizer == "adam":
            self.estimation_optimizer = Adam(learning_rate=lr, clipnorm=1.0)
            self.improvement_optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        else:
            raise ValueError("invalid optimizer name")

        self.metric_ce_x = tf.keras.metrics.Mean(name="ce_x")
        self.metric_ce_y = tf.keras.metrics.Mean(name="ce_y")

    def sample_inputs(self, batch, N):
        if self.input_distribution == 'sc' or self.input_distribution == 'sct':
            batch_N_shape = [tf.constant(batch), tf.constant(N)]
            ex_enc = self.Ex_enc.call(batch_N_shape)

            # generate shared randomness
            r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)
            # create frozen bits for encoding. encoded bits need to be 0.5.
            f_enc = 0.5 * tf.ones(shape=(batch, N, 1))
            _, x, llr_x = self.encode(ex_enc, f_enc, N, r, sample=True)
        elif self.input_distribution == 'rnn':
            x, llr_x = self.input_model.generate_binary_sequence(batch, N)
        else:
            x, llr_x = None, None
            ValueError(f'input_distribution received invalid value: {self.input_distribution}')
        return x, llr_x

    @tf.function
    def estimation_step(self, batch, N, train_ex=False):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]

        x, _ = self.sample_inputs(batch, N)
        y = self.channel.sample_channel_outputs(x)

        with tf.GradientTape(persistent=True) as tape:
            ex = self.Ex(batch_N_shape)
            ey = self.Ey(y)

            loss_x_array, _ = self.fast_ce(ex, x)
            loss_y_array, _ = self.fast_ce(ex+ey, x)

            loss = tf.reduce_mean(loss_x_array + loss_y_array)

        # Compute gradients
        trainable_vars = self.checknode.trainable_weights + \
                         self.bitnode.trainable_weights + \
                         self.emb2llr.trainable_weights + \
                         self.Ey.trainable_weights
        if train_ex:
            trainable_vars += self.Ex.trainable_weights

        gradients = tape.gradient(loss, trainable_vars)
        self.estimation_optimizer.apply_gradients(zip(gradients, trainable_vars))

        ce_x = tf.reduce_mean(loss_x_array[:, -1, :]) / tf.math.log(2.0)
        ce_y = tf.reduce_mean(loss_y_array[:, -1, :]) / tf.math.log(2.0)
        self.metric_ce_x.update_state(ce_x)
        self.metric_ce_y.update_state(ce_y)

        # Return a dict mapping metric names to current value
        res = {self.metric_ce_x.name: self.metric_ce_x.result(),
               self.metric_ce_y.name: self.metric_ce_y.result(),
               'loss': loss}
        return res

    @tf.function
    def improvement_step(self, batch, N):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]

        with tf.GradientTape() as tape:
            x, llr_x1 = self.sample_inputs(batch, N)
            llr_x = tf.where(tf.equal(x, 1.0), llr_x1, -llr_x1)
            log_px = tf.math.log(tf.math.sigmoid(llr_x)+1e-10)
            y = self.channel.sample_channel_outputs(x)

            ey = self.Ey(y)
            ex = self.Ex(batch_N_shape)

            loss_x_array, _ = self.fast_ce(ex, x)
            loss_y_array, _ = self.fast_ce(ex+ey, x)

            loss_x = tf.reduce_mean(loss_x_array[:, -1, :], axis=1)
            loss_y = tf.reduce_mean(loss_y_array[:, -1, :], axis=1)
            loss = -tf.reduce_mean(tf.stop_gradient(loss_x - loss_y - tf.reduce_mean(loss_x - loss_y)) *
                                   tf.reduce_mean(log_px, axis=(1, 2)))

        # Compute gradients
        if self.input_distribution == 'sc' or self.input_distribution == 'sct':
            trainable_vars = self.Ex_enc.trainable_weights
        elif self.input_distribution == 'rnn':
            trainable_vars = self.input_model.trainable_weights
        else:
            raise ValueError(f'input_distribution received invalid value: {self.input_distribution}')

        gradients = tape.gradient(loss, trainable_vars)
        self.improvement_optimizer.apply_gradients(zip(gradients, trainable_vars))

        ce_x = tf.reduce_mean(loss_x_array[:, -1, :]) / tf.math.log(2.0)
        ce_y = tf.reduce_mean(loss_y_array[:, -1, :]) / tf.math.log(2.0)
        self.metric_ce_x.update_state(ce_x)
        self.metric_ce_y.update_state(ce_y)

        # Return a dict mapping metric names to current value
        res = {self.metric_ce_x.name: self.metric_ce_x.result(),
               self.metric_ce_y.name: self.metric_ce_y.result(),
               'loss': loss,
               'px': tf.reduce_mean(llr_x1)}
        return res

    def train(self, train_block_length=8, train_batch=10, num_iters=10000,
              logging_freq=1000, saving_freq=3600, train_ex=False, load_nsc_path=None, save_model=False):

        if load_nsc_path is not None:
            self.load_model(load_nsc_path)

        save_name = f'train_' \
                    f'group-{wandb.run.group}_' \
                    f'{self.input_distribution_name}_' \
                    f'{self.channel.save_name}_' \
                    f'nt-{train_block_length}_' \
                    f'npd-{self.embedding_size}-{self.layers_per_op}x{self.hidden_size}'

        self.reset_metrics()

        t_save = time()
        t = time()
        for k in range(num_iters):
            r = self.estimation_step(train_batch, train_block_length, train_ex=train_ex)

            if k % logging_freq == 0:
                wandb.log({"iter_scl": k,
                           "ce_x": r["ce_x"].numpy(),
                           "ce_y": r["ce_y"].numpy(),
                           "mi": (r["ce_x"].numpy() - r["ce_y"].numpy()),
                           "loss": r["loss"].numpy()})
                print(f"iterations: {k + 1}, "
                      f"ce_x: {r['ce_x']: 6.5f} "
                      f"ce_y: {r['ce_y']: 6.5f} "
                      f"mi: {(r['ce_x'].numpy() - r['ce_y'].numpy()): 6.5f} "
                      f"loss: {r['loss']: 6.5f} "
                      f"elapsed: {time() - t: 4.3f}")
                t = time()

            if time() - t_save > saving_freq:
                self.reset_metrics()
                print("reset metrics")
                if save_model:
                    self.save_model(save_name)
                t_save = time()

        if save_model:
            self.save_model(save_name)

    def optimize(self, train_block_length=8, train_batch=10, num_iters=10000,
                 logging_freq=1000, saving_freq=3600, load_nsc_path=None, save_model=False):
        if load_nsc_path is not None:
            self.load_model(load_nsc_path)

        save_name = f'optimize_group-{wandb.run.group}_{self.input_distribution_name}_{self.channel.save_name}_nt-{train_block_length}_' \
                    f'npd-{self.embedding_size}-{self.layers_per_op}x' \
                    f'{self.hidden_size}'

        self.reset_metrics()

        t_save = time()
        t = time()
        for k in range(num_iters):
            _ = self.estimation_step(train_batch, train_block_length, train_ex=True)
            r = self.improvement_step(train_batch, train_block_length)

            if k % logging_freq == 0:
                wandb.log({"iter_scl": k,
                           "ce_x": r["ce_x"].numpy(),
                           "ce_y": r["ce_y"].numpy(),
                           "mi": (r["ce_x"].numpy() - r["ce_y"].numpy()),
                           "loss": r["loss"].numpy()})
                print(f"iterations: {k + 1}, "
                      f"ce_x: {r['ce_x']: 6.5f} "
                      f"ce_y: {r['ce_y']: 6.5f} "
                      f"mi: {(r['ce_x'].numpy() - r['ce_y'].numpy()): 6.5f} "
                      f"loss: {r['loss']: 4.3e} "
                      f"px: {r['px']: 6.5f}",
                      f"lr: {self.improvement_optimizer.learning_rate.numpy(): 6.5e}",
                      f"elapsed: {time() - t: 4.3f}")
                t = time()
                self.estimation_optimizer.learning_rate.assign(self.estimation_optimizer.learning_rate * 0.9995)
                self.improvement_optimizer.learning_rate.assign(self.improvement_optimizer.learning_rate * 0.9995)

            if time() - t_save > saving_freq:
                print("reset metrics...")
                self.reset_metrics()
                if save_model:
                    self.save_model(save_name)
                t_save = time()

        if save_model:
            self.save_model(save_name)

    def save_model(self, save_name):
        try:
            save_tmp_path = f"./artifacts/tmp/{wandb.run.name}/design.npy"
            # Extract the directory part of the path
            directory = os.path.dirname(save_tmp_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            weights_artifact = wandb.Artifact(save_name, type='model_weights')

            self.checknode.save_weights(f"{directory}/checknode.weights.h5")
            self.bitnode.save_weights(f"{directory}/bitnode.weights.h5")
            self.emb2llr.save_weights(f"{directory}/emb2llr.weights.h5")
            self.Ey.save_weights(f"{directory}/ey.weights.h5")
            self.Ex.save_weights(f"{directory}/ex.weights.h5")

            # Add the model weights file to the artifact
            weights_artifact.add_file(f"{directory}/checknode.weights.h5")
            weights_artifact.add_file(f"{directory}/bitnode.weights.h5")
            weights_artifact.add_file(f"{directory}/emb2llr.weights.h5")
            weights_artifact.add_file(f"{directory}/ey.weights.h5")
            weights_artifact.add_file(f"{directory}/ex.weights.h5")

            if self.input_distribution == 'sc' or self.input_distribution == 'sct':
                self.Ex_enc.save_weights(f"{directory}/ex_enc.weights.h5")
                weights_artifact.add_file(f"{directory}/ex_enc.weights.h5")
            elif self.input_distribution == 'rnn':
                self.input_model.save_weights(f"{directory}/input_model.weights.h5")
                weights_artifact.add_file(f"{directory}/input_model.weights.h5")
            else:
                raise ValueError(f'input_distribution received invalid value: {self.input_distribution}')

            # Log the artifact to W&B
            artifact_log = wandb.log_artifact(weights_artifact)
            artifact_log.wait()
            # print(f"model is saved as {save_name}")
            artifact = wandb.run.use_artifact(f"{save_name}:latest")
            print(f"Artifact: {artifact.source_name}")
        except Exception as e:
            print(f"An error occurred: {e}")


class SCListDecoder(SCDecoder):
    def __init__(self, channel, batch=100, list_num=4, crc=None, crc_oracle=None, *args, **kwargs):
        SCDecoder.__init__(self, channel, batch, *args, **kwargs)
        self.nL = list_num
        self.crc = crc
        if self.crc is not None:
            self.crc_enc = CRCEncoder(crc_degree=crc)
            self.crc_dec = CRCDecoder(crc_encoder=self.crc_enc)
            self.crc_oracle = crc_oracle

        self.checknode_list = CheckNodeVanilla()
        self.bitnode_list = BitNodeVanilla()
        self.emb2llr_list = Activation(tf.identity)
        self.split_even_odd_list = SplitEvenOdd(axis=2)
        self.f2_list = F2(axis=2)
        self.interleave_list = Interleave(axis=2)

    @tf.function
    def decode_list(self, ex, ey, f, pm, N, r, sample=True):
        if N == 1:
            llr_u = self.emb2llr_list(ex)
            if sample:
                pu = tf.math.sigmoid(llr_u)
                frozen = tf.where(tf.greater_equal(r, pu), 0.0, 1.0)
            else:
                frozen = self.hard_decision(llr_u)

            dm = self.emb2llr_list.call(ey)
            hd_ = self.hard_decision.call(tf.squeeze(dm, axis=(2, 3)))
            hd = tf.concat((hd_, 1 - hd_), axis=1)
            pm_dup = tf.concat((pm, pm + tf.abs(tf.squeeze(dm, axis=(2, 3)))), -1)
            pm_prune, prune_idx_ = tf.math.top_k(-pm_dup, k=self.nL, sorted=True)
            pm_prune = -pm_prune
            prune_idx = tf.sort(prune_idx_, axis=1)
            idx = tf.argsort(prune_idx_, axis=1)
            pm_prune = tf.gather(pm_prune, idx, axis=1, batch_dims=1)
            u_survived = tf.gather(hd, prune_idx, axis=1, batch_dims=1)[:, :, tf.newaxis, tf.newaxis]

            is_frozen = tf.not_equal(f, 0.5)
            x = tf.where(is_frozen, frozen, u_survived)
            pm_ = tf.where(tf.squeeze(is_frozen, axis=(2, 3)),
                           pm + tf.abs(tf.squeeze(dm, axis=(2, 3))) *
                           tf.cast(tf.squeeze(tf.not_equal(tf.expand_dims(tf.expand_dims(hd_, -1), -1), frozen),
                                              axis=(2, 3)), tf.float32),
                           pm_prune)
            new_order = tf.tile(tf.expand_dims(tf.range(self.nL), 0), [ey.shape[0], 1]) \
                if f[0, 0, 0, 0] != 0.5 else (prune_idx % self.nL)
            return x, tf.identity(x), llr_u, dm, pm_, new_order

        f_halves = tf.split(f, num_or_size_splits=2, axis=2)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=2)
        r_left, r_right = r_halves

        ey_odd, ey_even = self.split_even_odd_list.call(ey)
        ex_odd, ex_even = self.split_even_odd_list.call(ex)
        e_odd, e_even = tf.concat([ex_odd, ey_odd], axis=0), tf.concat([ex_even, ey_even], axis=0)

        # Compute soft mapping back one stage
        e1est = self.checknode_list.call((e_odd, e_even))
        ex1est, ey1est = tf.split(e1est, num_or_size_splits=2, axis=0)

        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, llr_u_left, llr_uy_left, pm, new_order = self.decode_list(ex1est, ey1est, f_left, pm,
                                                                                     N // 2, r_left, sample=sample)

        new_order_dup = tf.concat((new_order,
                                   new_order), axis=0)
        e_odd = tf.gather(e_odd, new_order_dup, axis=1, batch_dims=1)
        e_even = tf.gather(e_even, new_order_dup, axis=1, batch_dims=1)

        # Using u1est and x1hard, we can estimate u2
        u1_hardprev_dup = tf.tile(u1hardprev, [2, 1, 1, 1])
        e2est = self.bitnode_list.call((e_odd, e_even, u1_hardprev_dup))
        ex2est, ey2est = tf.split(e2est, num_or_size_splits=2, axis=0)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, llr_u_right, llr_uy_right, pm, new_order2 = self.decode_list(ex2est, ey2est, f_right, pm,
                                                                                        N // 2, r_right, sample=sample)
        uhat1 = tf.gather(uhat1, new_order2, axis=1, batch_dims=1)
        llr_u_left = tf.gather(llr_u_left, new_order2, axis=1, batch_dims=1)
        llr_uy_left = tf.gather(llr_uy_left, new_order2, axis=1, batch_dims=1)
        u1hardprev = tf.gather(u1hardprev, new_order2, axis=1, batch_dims=1)
        new_order = tf.gather(new_order, new_order2, axis=1, batch_dims=1)
        u = tf.concat([uhat1, uhat2], axis=2)
        llr_u = tf.concat([llr_u_left, llr_u_right], axis=2)
        llr_uy = tf.concat([llr_uy_left, llr_uy_right], axis=2)
        x = self.f2_list.call((u1hardprev, u2hardprev))
        x = self.interleave_list.call(x)
        return u, x, llr_u, llr_uy, pm, new_order

    # @tf.function
    def forward_eval(self, batch, N, info_indices, frozen_indices, A, Ac):
        batch_N_shape = [tf.constant(batch), tf.constant(N)]
        # generate the information bits:
        # bits = tf.cast(tf.random.uniform((batch, N), minval=0, maxval=2, dtype=tf.int32), dtype)
        #
        # # create frozen bits for encoding. encoded bits need to be 0.5. info bit 0/1
        # updates = 0.5 * tf.ones([batch * tf.shape(Ac)[0]], dtype=dtype)
        # f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(bits, frozen_indices, updates), axis=2)
        f_enc = tf.ones(shape=(batch, N)) * 0.5
        if self.crc is None:
            info_bits = tf.cast(tf.random.uniform((batch, tf.shape(A)[0],), minval=0, maxval=2, dtype=tf.int32), dtype)
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(info_bits, [-1])), axis=2)
        else:
            info_bits_num = len(A) - self.crc_enc.crc_length
            info_bits = tf.cast(tf.random.uniform((batch, info_bits_num,), minval=0, maxval=2, dtype=tf.int32), dtype)
            bits = self.crc_enc(info_bits)
            bits = tf.reverse(bits, axis=[-1])
            # bits = tf.transpose(bits)
            f_enc = tf.expand_dims(tf.tensor_scatter_nd_update(f_enc, info_indices, tf.reshape(bits, [-1])), axis=2)

        # generate shared randomness
        r = tf.random.uniform(shape=(batch, N, 1), dtype=tf.float32)

        # encode the bits into x^N and u^N
        ex_enc = self.Ex_enc.call(batch_N_shape)
        u, x, llr_u1_enc = self.encode(ex_enc, f_enc, N, r, sample=True)
        y = self.channel.sample_channel_outputs(x)

        # create frozen bits for decoding. decoded bits need to be 0.5.
        # frozen bits are 0 decoded using argmax like in the encoder
        tensor = tf.zeros(shape=(batch, N))
        updates = 0.5 * tf.ones([batch * tf.shape(A)[0]], dtype=dtype)
        f_dec = tf.expand_dims(tf.tensor_scatter_nd_update(tensor, info_indices, updates), axis=2)

        f_dec = tf.tile(tf.expand_dims(f_dec, 1), [1, self.nL, 1, 1])
        r = tf.tile(tf.expand_dims(r, 1), [1, self.nL, 1, 1])

        ey = self.Ey(y)
        ex_ = tf.expand_dims(ex_enc, 1)
        ey_ = tf.expand_dims(ex_enc+ey, 1)
        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(ex_)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.nL]))
        ex_dup = tf.tile(ex_, repmat)
        ey_dup = tf.tile(ey_, repmat)
        pm = tf.concat([tf.zeros([1]), tf.ones([self.nL-1])*float('inf')], 0)
        pm = tf.tile(tf.expand_dims(pm, 0), [u.shape[0], 1])
        uhat_list, xhat, llr_u, llr_uy, pm, new_order = self.decode_list(ex_dup, ey_dup, f_dec, pm,
                                                                         f_dec.shape[2], r, sample=True)

        if self.crc is None:
            uhat = self.choose_codeword_pm(uhat_list, pm)
        else:
            if self.crc_oracle:
                uhat = self.choose_codeword_oracle(uhat_list, u)
            else:
                uhat = self.choose_codeword_crc(uhat_list, pm, A)

        errors = tf.cast(tf.where(tf.equal(uhat, u), 0, 1), tf.float32)
        info_bit_errors = tf.gather(params=errors,
                                    indices=A,
                                    axis=1)
        return tf.squeeze(tf.reduce_mean(info_bit_errors, axis=1), axis=1)

    @staticmethod
    def choose_codeword_pm(uhat_list, pm):
        uhat = tf.gather(uhat_list, tf.argmin(pm, axis=1), axis=1, batch_dims=1)
        return uhat

    def choose_codeword_crc(self, uhat_list, pm, A):
        sort_ = tf.argsort(pm)
        uhat_list = tf.gather(params=uhat_list, indices=sort_, axis=1, batch_dims=1)
        crcs = []
        for i in range(self.nL):
            uhat_info_hat = tf.gather(params=uhat_list[:, i, :, 0],
                                      indices=tf.tile(tf.expand_dims(A, 0), [tf.shape(uhat_list)[0], 1]),
                                      batch_dims=1)
            _, crc_valid = self.crc_dec(tf.reverse(uhat_info_hat, axis=[-1]))
            crcs.append(crc_valid)
        crc_valid = tf.concat(crcs, axis=-1)
        chosen_idx = tf.argmax(crc_valid, axis=-1)
        uhat = tf.gather(uhat_list, indices=chosen_idx, batch_dims=1, axis=1)
        return uhat

    def choose_codeword_oracle(self, uhat_list, u):
        oracle_valid = tf.reduce_all(tf.equal(uhat_list,
                                              tf.tile(tf.expand_dims(u, 1), [1, self.nL, 1, 1])),
                                     axis=(2, 3))
        chosen_idx_oracle = tf.argmax(oracle_valid, axis=-1)
        uhat_oracle = tf.gather(uhat_list, indices=chosen_idx_oracle, batch_dims=1)
        return uhat_oracle

    def choose_information_and_frozen_sets(self, sorted_bit_channels, k):
        if self.crc is None:
            A = sorted_bit_channels[:k]
            A = sorted(A)
            Ac = sorted_bit_channels[k:]
            Ac = sorted(Ac)
        else:
            k_crc = self.crc_enc.crc_length
            A = sorted_bit_channels[:k+k_crc]
            A = sorted(A)
            Ac = sorted_bit_channels[k+k_crc:]
            Ac = sorted(Ac)
        return A, Ac


class SCTrellisListDecoder(SCListDecoder):
    def __init__(self, channel, batch=100, list_num=4, crc=None, *args, **kwargs):
        SCListDecoder.__init__(self, channel, batch, list_num=list_num, crc=crc, *args, **kwargs)

        self.llr_enc_shape = self.llr_dec_shape = (self.channel.cardinality_x,
                                                   self.channel.cardinality_s,
                                                   self.channel.cardinality_s)

        unnormalized_logits = tf.zeros(shape=(self.channel.cardinality_x,
                                              self.channel.cardinality_s,
                                              self.channel.cardinality_s))
        logits = unnormalized_logits - tf.math.reduce_logsumexp(unnormalized_logits, axis=(0, 2), keepdims=True)
        self.input_logits = tf.constant(logits, dtype=dtype)
        self.Ex = self.Ex_enc = EmbeddingX(self.input_logits)

        self.checknode_enc = self.checknode = CheckNodeTrellis(state_size=self.channel.cardinality_s)
        self.bitnode_enc = self.bitnode = BitNodeTrellis(state_size=self.channel.cardinality_s)
        self.emb2llr_enc = self.emb2llr = Embedding2LLRTrellis()
        self.checknode_list = CheckNodeTrellis(state_size=self.channel.cardinality_s, batch_dims=3)
        self.bitnode_list = BitNodeTrellis(state_size=self.channel.cardinality_s, batch_dims=3)
        self.emb2llr_list = Embedding2LLRTrellis(batch_dims=3)


class SCNeuralListDecoder(SCListDecoder, SCNeuralDecoder):

    def __init__(self, channel, embedding_size, hidden_size, layers_per_op,
                 activation='elu', batch=100, list_num=4, crc=None, *args, **kwargs):
        SCListDecoder.__init__(self, channel=channel, batch=batch, list_num=list_num, crc=crc, *args, **kwargs)
        SCNeuralDecoder.__init__(self, channel=channel, batch=batch, embedding_size=embedding_size,
                                 hidden_size=hidden_size, layers_per_op=layers_per_op, activation=activation, *args, **kwargs)
        self.checknode_list = self.checknode
        self.bitnode_list = self.bitnode
        self.emb2llr_list = self.emb2llr
