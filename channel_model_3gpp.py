import sionna as sn
import tensorflow as tf
import numpy as np
from sionna.channel.tr38901 import TDL
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.python.keras import initializers

from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder, PolarSCDecoder
from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code

tf.keras.backend.set_floatx('float32')
# Define the number of UT and BS antennas
NUM_UT = 1
NUM_BS = 1
NUM_UT_ANT = 1
NUM_BS_ANT = 1
dtype = tf.keras.backend.floatx()


def bpsk_modulation(bits):
    """
    Perform BPSK modulation on a sequence of bits.

    Args:
        bits (tf.Tensor): A tensor of shape (n,) containing binary bits (0s and 1s).

    Returns:
        tf.Tensor: A tensor of shape (n,) containing BPSK modulated values (+1s and -1s).
    """
    # Ensure the input is a TensorFlow tensor
    # bits = tf.convert_to_tensor(bits, dtype=tf.float32)

    # Perform BPSK modulation: map 0 -> +1 and 1 -> -1
    bpsk_signal = 1 - 2 * bits

    return bpsk_signal


class OFDMSystem(Model):  # Inherits from Keras Model

    def __init__(self, cfg, snrdb=(6.0, 6.0), No_pilots=False):
        super().__init__()
        # init params




        self.snrdb = snrdb
        self.cfg = cfg
        self.save_name = f"5g3gpp_snrl-{snrdb[0]}_snrh-{snrdb[1]}"
        self.cardinality_x = self.cardinality_s = 2
        self.cardinality_y = 2
        self.perfect_csi = cfg['perfect_csi']
        self.BPS = cfg['NUM_BITS_PER_SYMBOL']
        self.CODERATE = cfg['code_rate']
        num_ofdm_symbols = cfg['num_ofdm_symbols']
        subcarrier_spacing = cfg['subcarrier_spacing']
        cyclic_prefix_length = 0  #cfg['cyclic_prefix_length']  # is not importent for freq simulation
        channel_mode = cfg['channel_mode']
        delay_spread = cfg['delay_spread']
        carrier_frequency = cfg['carrier_frequency']
        fft_size = cfg['subcarrier_num']

        pilot_ofdm_symbol_indices = np.linspace(1, num_ofdm_symbols + int(self.cfg['num_pilots']) - 2,
                                                int(self.cfg['num_pilots']))
        pilot_ofdm_symbol_indices = [int(i) for i in pilot_ofdm_symbol_indices]
        num_ofdm_symbols_with_pilots = num_ofdm_symbols + len(pilot_ofdm_symbol_indices)
        self.RESOURCE_GRID = sn.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols_with_pilots,
                                                  fft_size=fft_size,
                                                  subcarrier_spacing=subcarrier_spacing,
                                                  num_tx=1,
                                                  num_streams_per_tx=1,
                                                  cyclic_prefix_length=cyclic_prefix_length,
                                                  pilot_pattern="kronecker",
                                                  pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        self.rg = self.RESOURCE_GRID

        if No_pilots:
            self.RESOURCE_GRID_no_pilots = sn.ofdm.ResourceGrid(num_ofdm_symbols=num_ofdm_symbols,
                                                                fft_size=fft_size,
                                                                subcarrier_spacing=subcarrier_spacing,
                                                                num_tx=1,
                                                                num_streams_per_tx=1,
                                                                cyclic_prefix_length=cyclic_prefix_length,
                                                                pilot_pattern="empty",
                                                                pilot_ofdm_symbol_indices=None)
            self.rg = self.RESOURCE_GRID_no_pilots

        # The mapper maps blocks of information bits to constellation symbols
        if self.BPS == 1:
            bpsk_constellation = sn.mapping.Constellation("custom", 1,
                                                          initial_value=tf.convert_to_tensor([1, -1]))
            self.mapper = sn.mapping.Mapper(constellation_type="custom",
                                            num_bits_per_symbol=self.BPS,
                                            constellation=bpsk_constellation)
        else:
            self.mapper = sn.mapping.Mapper("qam", self.BPS)

        # OFDM modulator
        self.rg_mapper = sn.ofdm.ResourceGridMapper(self.rg)
        self.OFDMModulator = sn.ofdm.OFDMModulator(0)

        # Create a channel model
        if self.cfg['channel_type'] == 'TDL':
            self.AWGN = False
            tdl = TDL(model=channel_mode,
                      delay_spread=delay_spread,
                      carrier_frequency=carrier_frequency,
                      min_speed=0.0,
                      max_speed=3.0)
            # Frequency domain channel
            self.channel = sn.channel.OFDMChannel(tdl,
                                                  self.rg,
                                                  add_awgn=True,
                                                  normalize_channel=True,
                                                  return_channel=True)

        # channel estimate
        # The LS channel estimator will provide channel estimates and error variances
        self.ls_est = sn.ofdm.LSChannelEstimator(self.RESOURCE_GRID, interpolation_type="nn")

        # The LMMSE equalizer will provide soft symbols together with noise variance estimates
        NUM_STREAMS_PER_TX = NUM_UT_ANT
        RX_TX_ASSOCIATION = np.array([[1]])
        STREAM_MANAGEMENT = sn.mimo.StreamManagement(RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX)
        self.lmmse_equ = sn.ofdm.LMMSEEqualizer(self.RESOURCE_GRID, STREAM_MANAGEMENT)

        # The demapper produces LLR for all coded bits
        if self.BPS == 1:
            self.demapper = sn.mapping.Demapper("app", "custom", self.BPS, constellation=bpsk_constellation)
        else:
            self.demapper = sn.mapping.Demapper("app", "qam", self.BPS)

        self.h_freq = self.add_weight(name="state", shape=(cfg['batch'], 1, 1, 1, 1, num_ofdm_symbols_with_pilots, fft_size), dtype=tf.complex64, trainable=False,
                                     initializer=initializers.constant(0.0))

        self.y = self.add_weight(name="state", shape=(cfg['batch'], 1, 1, num_ofdm_symbols_with_pilots, fft_size), dtype=tf.complex64, trainable=False,
                                     initializer=initializers.constant(0.0))
        self.no = self.add_weight(name="state", shape=(1), dtype=dtype, trainable=False,
                                     initializer=initializers.constant(0.0))

    def llr(self, y):
        no = self.no[0]
        if self.perfect_csi:
            h_hat, err_var = self.h_freq, 0.
        else:
            h_hat, err_var = self.ls_est([self.y, no])

        x_hat, no_eff = self.lmmse_equ([self.y, h_hat, err_var, no])
        llr = self.demapper([x_hat, no_eff])
        llr = tf.squeeze(llr, axis=(1, 2))[:, :, tf.newaxis]
        #llr = tf.concat([tf.math.real(llr), tf.math.imag(llr)], axis=-1)
        return llr
        #raise NotImplementedError

    def generate_ebno(self):
        ebno_db = tf.random.uniform(shape=(), minval=self.snrdb[0], maxval=self.snrdb[1], dtype=tf.float32)
        ebno = tf.math.pow(tf.cast(10., dtype), ebno_db / 10.)
        no = 1/ebno
        return no
        # return sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.BPS, coderate=self.CODERATE,
        #                           resource_grid=self.RESOURCE_GRID)

    # @tf.function  # Graph execution to speed things up
    def sample_channel_outputs(self, codewords):
        no = self.generate_ebno()
        self.no.assign([no])
        # self.no.assign(h_freq)

        codewords = tf.squeeze(codewords, axis=2)
        codewords = codewords[:, tf.newaxis, tf.newaxis, :]
        x_mapper = self.mapper(codewords)
        x_rg = self.rg_mapper(x_mapper)

        # Channel
        y_complex, h_freq = self.channel([x_rg, no])
        self.h_freq.assign(h_freq)
        #
        self.y.assign(y_complex)
        y_complex = tf.reshape(y_complex, tf.concat((y_complex.shape[:3], [-1]), axis=0))
        y_complex = tf.squeeze(y_complex)[:, :, tf.newaxis]

        y_real = tf.concat([tf.math.real(y_complex), tf.math.imag(y_complex)], axis=-1)
        return y_real
