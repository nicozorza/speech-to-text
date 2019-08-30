import math
from typing import List
import tensorflow as tf
from src.neural_network.network_utils import bidirectional_lstm, lstm_cell
import numpy as np

def bidirectional_pyramidal_rnn(input_ph, seq_len_ph, num_layers: int, num_units: List[int], name: str, activation_list,
                                use_tensorboard: bool = True, tensorboard_scope: str = None, keep_prob=None, train_ph=False,
                                keep_state: bool = False):

    if activation_list is None:
        activation_list = [None] * num_layers

    if keep_prob is None:
        keep_prob = [None] * num_layers

    initial_state_fw = None
    initial_state_bw = None
    state = None

    for n in range(num_layers):
        inputs, state = bidirectional_lstm(
            input_ph=input_ph,
            seq_len_ph=seq_len_ph,
            num_units=num_units[n],
            scope=name+'_{}'.format(n),
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            activation=activation_list[n],
            keep_prob=keep_prob[n],
            train_ph=train_ph
        )

        inputs = tf.concat(inputs, -1)
        input_ph, seq_len_ph = reshape_pyramidal(inputs, seq_len_ph)
        if keep_state:
            initial_state_fw, initial_state_bw = state

        if use_tensorboard:
            tf.summary.histogram(tensorboard_scope, input_ph)

    return input_ph, seq_len_ph, state


def reshape_pyramidal(outputs, sequence_length):
    # [batch_size, max_time, num_units]
    shape = tf.shape(outputs)
    batch_size, max_time = shape[0], shape[1]
    num_units = outputs.get_shape().as_list()[-1]

    pads = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
    outputs = tf.pad(outputs, pads)

    concat_outputs = tf.reshape(outputs, (batch_size, -1, num_units * 2))
    return concat_outputs, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)


def scaled_dot_product(input_ph, input_len, hidden_dim, output_dim, scaled=True, masked=True,
                       activation=None, name=None, use_tensorboard=True):
    Q = tf.layers.dense(
        input_ph,
        activation=activation,
        units=hidden_dim,
        name=name + "_q" if name is not None else None)  # [batch_size, sequence_length, hidden_dim]
    K = tf.layers.dense(
        input_ph,
        activation=activation,
        units=hidden_dim,
        name=name + "_k" if name is not None else None)  # [batch_size, sequence_length, hidden_dim]
    V = tf.layers.dense(
        input_ph,
        activation=activation,
        units=output_dim,
        name=name + "_v" if name is not None else None)  # [batch_size, sequence_length, n_classes]

    if use_tensorboard:
        tf.summary.histogram(Q.name, Q)
        tf.summary.histogram(K.name, K)
        tf.summary.histogram(V.name, V)

    attention = tf.matmul(Q, K, transpose_b=True)  # [batch_size, sequence_length, sequence_length]

    if scaled:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]

    if masked:
        multiplies = tf.shape(input_ph)[1]
        mask = tf.sequence_mask(
            tf.tile(tf.expand_dims(input_len, axis=-1), [1, multiplies]),
            maxlen=tf.reduce_max(input_len),
            dtype=tf.float32)
        attention = tf.multiply(attention, mask)

    attention = tf.nn.softmax(attention, axis=-1)  # [batch_size, sequence_length, sequence_length]

    output = tf.matmul(attention, V)  # [batch_size, sequence_length, output_dim]
    return output


def self_attention(input_ph, input_len, hidden_dim, output_dim, scaled=True, masked=True, activation=None):
    return scaled_dot_product(input_ph=input_ph,
                              input_len=input_len,
                              hidden_dim=hidden_dim,
                              output_dim=output_dim,
                              scaled=scaled,
                              masked=masked,
                              activation=activation)


def multihead_attention(input_ph, input_len, num_heads, hidden_dim, hidden_output, output_dim, scaled=True,
                        masked=True, activation=None):
    head_list = []

    for i in range(num_heads):
        head_list.append(scaled_dot_product(    # [batch_size, sequence_length, output_dim]
            input_ph=input_ph,
            input_len=input_len,
            hidden_dim=hidden_dim,
            output_dim=hidden_output,
            scaled=scaled,
            masked=masked,
            activation=activation,
            name=f"head_{i}")
        )
    attention = tf.concat(head_list, axis=2)    # [batch_size, sequence_length, output_dim * num_heads]

    output = tf.layers.dense(attention, output_dim)  # [batch_size, sequence_length, output_dim]

    return output


def attention_cell(input, lengths, num_layers: int, attention_units: int, attention_size: int, attention_type: str,
                   activation, keep_prob, train_ph, use_tensorboard=True, tensorboard_scope='attention_cell'):

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(attention_units, activation, keep_prob, train_ph) for _ in range(num_layers)])

    if attention_type is 'luong':
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=attention_units,
                                                                memory=input,
                                                                memory_sequence_length=lengths)
    elif attention_type is 'bahdanau':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=attention_units,
                                                                   memory=input,
                                                                   memory_sequence_length=lengths)
    else:
        raise ValueError('Invalid attention mechanism')

    if use_tensorboard:
        tf.summary.histogram(tensorboard_scope, input)

    return tf.contrib.seq2seq.AttentionWrapper(cell=cell,
                                               attention_mechanism=attention_mechanism,
                                               alignment_history=not train_ph,
                                               attention_layer_size=attention_size,
                                               output_attention=attention_type is 'luong')


def attention_layer(input, lengths, num_layers: int, attention_units: int, activation, attention_size,
                    attention_type: str, batch_size, keep_prob, train_ph, input_state=None,
                    use_tensorboard=True, tensorboard_scope='attention_cell'):
    cell = attention_cell(input=input,
                          lengths=lengths,
                          num_layers=num_layers,
                          attention_units=attention_units,
                          attention_size=attention_size,
                          attention_type=attention_type,
                          activation=activation,
                          keep_prob=keep_prob,
                          train_ph=train_ph,
                          use_tensorboard=use_tensorboard,
                          tensorboard_scope=tensorboard_scope)

    state = cell.zero_state(batch_size, tf.float32)
    if input_state is not None:
        state = state.clone(cell_state=input_state)
    return cell, state


def attention_decoder(input_cell, initial_state, embedding_fn, seq_embedding,
                      seq_embedding_len, projection_layer, sampling_prob):

    if sampling_prob > 0.0:
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            seq_embedding, seq_embedding_len,
            embedding_fn, sampling_prob)
    else:
        helper = tf.contrib.seq2seq.TrainingHelper(
            seq_embedding, seq_embedding_len)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        input_cell, helper, initial_state, output_layer=projection_layer)

    return decoder


def beam_search_decoder(input_cell, embedding, start_token, end_token, initial_state,
                        beam_width, projection_layer, batch_size):
    start_token = tf.fill([tf.div(batch_size, beam_width)], start_token)

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=input_cell,
        embedding=embedding,
        start_tokens=start_token,
        end_token=end_token,
        initial_state=initial_state,
        beam_width=beam_width,
        output_layer=projection_layer)

    return decoder


def greedy_decoder(inputs, embedding, start_token, end_token,
                   initial_state, projection_layer, batch_size):

    start_token = tf.fill([batch_size], start_token)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_token, end_token)

    decoder = tf.contrib.seq2seq.BasicDecoder(inputs, helper, initial_state,
                                              output_layer=projection_layer)

    return decoder


def attention_loss(logits, targets, logits_length, targets_length, eos_id, train_ph):

    if train_ph:
        target_weights = tf.sequence_mask(targets_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, target_weights)
    else:
        '''
        # Reference: https://github.com/WindQAQ/listen-attend-and-spell
        '''

        max_ts = tf.reduce_max(targets_length)
        max_fs = tf.reduce_max(logits_length)

        max_sequence_length = tf.to_int32(tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])]], constant_values=eos_id)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max([targets_length, logits_length], 0)

        target_weights = tf.sequence_mask(sequence_length, maxlen=max_sequence_length, dtype=tf.float32)

        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, target_weights)

    return loss


def dense_to_sparse(tensor, eos_id, merge_repeated=True):
    if merge_repeated:
        added_values = tf.cast(tf.fill((tf.shape(tensor)[0], 1), eos_id), tensor.dtype)

        # merge consecutive values
        concat_tensor = tf.concat((tensor, added_values), axis=-1)
        diff = tf.cast(concat_tensor[:, 1:] - concat_tensor[:, :-1], tf.bool)

        # trim after first eos token
        eos_indices = tf.where(tf.equal(concat_tensor, eos_id))
        first_eos = tf.segment_min(eos_indices[:, 1], eos_indices[:, 0])
        mask = tf.sequence_mask(first_eos, maxlen=tf.shape(tensor)[1])

        indices = tf.where(diff & mask & tf.not_equal(tensor, -1))
        values = tf.gather_nd(tensor, indices)
        shape = tf.shape(tensor, out_type=tf.int64)

        return tf.SparseTensor(indices, values, shape)
    else:
        return tf.contrib.layers.dense_to_sparse(tensor, eos_id)


def edit_distance(hypothesis, truth, eos_id, mapping=None):

    if mapping:
        mapping = tf.convert_to_tensor(mapping)
        hypothesis = tf.nn.embedding_lookup(mapping, hypothesis)
        truth = tf.nn.embedding_lookup(mapping, truth)

    hypothesis = dense_to_sparse(hypothesis, eos_id, merge_repeated=True)
    truth = dense_to_sparse(truth, eos_id, merge_repeated=True)

    return tf.edit_distance(hypothesis, truth, normalize=True)


def get_positional_encoding_mask(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0):

    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


# Obtained from http://jalammar.github.io/illustrated-transformer/
def add_positional_encoding(input_ph, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    # [batch_size, sequence_length, num_features]
    length = tf.shape(input_ph)[1]
    channels = tf.shape(input_ph)[2]
    signal = get_positional_encoding_mask(length, channels, min_timescale, max_timescale, start_index)
    return input_ph + signal
