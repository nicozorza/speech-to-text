from typing import List
import tensorflow as tf
from src.neural_network.network_utils import bidirectional_lstm, lstm_cell


def bidirectional_pyramidal_rnn(input_ph, seq_len_ph, num_layers: int, num_units: List[int], name: str, activation_list,
                                use_tensorboard: bool = True, tensorboard_scope: str = None, keep_prob=None, train_ph=False,
                                keep_state: bool = False):

    if activation_list is None:
        activation_list = [None] * num_layers

    if keep_prob is None:
        keep_prob = [None] * num_layers

    initial_state_fw = None
    initial_state_bw = None

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


def attention_cell(input, num_layers: int, rnn_units_list: List[int], rnn_activations_list,
                   attention_units, lengths, keep_prob, train_ph):

    if keep_prob is None:
        keep_prob = [None] * num_layers

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(rnn_units_list[_], rnn_activations_list[_], keep_prob[_], train_ph) for _ in range(num_layers)])

    # TODO Add other mechanisms
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=attention_units,
                                                               memory=input,
                                                               memory_sequence_length=lengths,
                                                               name='BahdanauAttention')

    return tf.contrib.seq2seq.AttentionWrapper(cell=cell,
                                               attention_mechanism=attention_mechanism,
                                               attention_layer_size=None,
                                               output_attention=False)


def attention_layer(input, num_layers: int, rnn_units_list: List[int], rnn_activations_list,
                    attention_units, lengths, batch_size, keep_prob, train_ph, input_state=None):
    cell = attention_cell(input=input,
                          num_layers=num_layers,
                          rnn_units_list=rnn_units_list,
                          rnn_activations_list=rnn_activations_list,
                          attention_units=attention_units,
                          lengths=lengths,
                          keep_prob=keep_prob,
                          train_ph=train_ph)

    state = cell.zero_state(batch_size, tf.float32)
    if input_state is not None:
        state = state.clone(cell_state=input_state)
    return cell, state


def attention_decoder(input_cell, initial_state, embedding, seq_embedding, seq_embedding_len,
                      output_projection, max_iterations, sampling_prob, time_major, name):
    # TODO Ver tf.contrib.legacy_seq2seq.attention_decoder para ver si es mejor que hacerlo a mano
    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        inputs=seq_embedding,
        sequence_length=seq_embedding_len,
        embedding=embedding,
        sampling_probability=sampling_prob,
        time_major=time_major)

    decoder = tf.contrib.seq2seq.BasicDecoder(cell=input_cell,
                                              helper=helper,
                                              initial_state=initial_state,
                                              output_layer=output_projection)

    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=time_major,
        maximum_iterations=max_iterations,
        swap_memory=False,
        impute_finished=True,
        scope=name
    )

    sample_id = outputs.sample_id
    logits = outputs.rnn_output

    return logits, sample_id, final_context_state


def beam_search_decoder(input_cell, embedding, start_token, end_token, initial_state,
                        beam_width, output_layer, max_iterations, name, time_major=False):
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=input_cell,
                                                   embedding=embedding,
                                                   start_tokens=start_token,
                                                   end_token=end_token,
                                                   initial_state=initial_state,
                                                   beam_width=beam_width,
                                                   output_layer=output_layer)

    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                        maximum_iterations=max_iterations,
                                                                        output_time_major=time_major,
                                                                        impute_finished=False,
                                                                        swap_memory=False,
                                                                        scope=name)
    return outputs.predicted_ids


def greedy_decoder(input_cell, embedding, start_token, end_token, initial_state, output_layer,
                   max_iterations, name, time_major=False):
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_token, end_token)

    decoder = tf.contrib.seq2seq.BasicDecoder(input_cell, helper, initial_state,
                                              output_layer=output_layer)

    outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                        maximum_iterations=max_iterations,
                                                                        output_time_major=time_major,
                                                                        impute_finished=False,
                                                                        swap_memory=False,
                                                                        scope=name)
    return outputs.sample_id


def attention_loss(logits, targets, final_sequence_length, target_sequence_length, eos_id, train_ph):

    if train_ph:
        target_weights = tf.sequence_mask(target_sequence_length, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, target_weights)
    else:
        '''
        # Reference: https://github.com/WindQAQ/listen-attend-and-spell
        '''

        max_ts = tf.reduce_max(target_sequence_length)
        max_fs = tf.reduce_max(final_sequence_length)

        max_sequence_length = tf.to_int32(tf.maximum(max_ts, max_fs))

        logits = tf.slice(logits, begin=[0, 0, 0], size=[-1, max_fs, -1])

        # pad EOS to make targets and logits have same shape
        targets = tf.pad(targets, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(targets)[1])]], constant_values=eos_id)
        logits = tf.pad(logits, [[0, 0], [0, tf.maximum(
            0, max_sequence_length - tf.shape(logits)[1])], [0, 0]], constant_values=0)

        # find larger length between predictions and targets
        sequence_length = tf.reduce_max([target_sequence_length, final_sequence_length], 0)

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
