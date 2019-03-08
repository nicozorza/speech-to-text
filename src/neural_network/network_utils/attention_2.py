import tensorflow as tf

from src.neural_network.network_utils import reshape_pyramidal, lstm_cell


# def lstm_cell(num_units, dropout, mode):
#     cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
#
#     dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 1.0
#
#     if dropout > 0.0:
#         cell = tf.nn.rnn_cell.DropoutWrapper(
#             cell=cell, input_keep_prob=dropout)
#
#     return cell


def bilstm(inputs,
           sequence_length,
           num_units,
           dropout,
           mode):

    with tf.variable_scope('fw_cell'):
        forward_cell = lstm_cell(size=num_units, activation=None,
                                 keep_prob=dropout, train_ph=mode == tf.estimator.ModeKeys.TRAIN)
    with tf.variable_scope('bw_cell'):
        backward_cell = lstm_cell(size=num_units, activation=None,
                                  keep_prob=dropout, train_ph=mode == tf.estimator.ModeKeys.TRAIN)

    return tf.nn.bidirectional_dynamic_rnn(
        forward_cell,
        backward_cell,
        inputs,
        sequence_length=sequence_length,
        dtype=tf.float32)


def pyramidal_bilstm(inputs,
                     sequence_length,
                     mode,
                     num_units,
                     keep_prob,
                     num_layers
                     ):

    outputs = inputs

    for layer in range(num_layers):
        with tf.variable_scope('bilstm_{}'.format(layer)):
            outputs, state = bilstm(
                outputs, sequence_length, num_units, keep_prob, mode)

            outputs = tf.concat(outputs, -1)

            if layer != 0:
                outputs, sequence_length = reshape_pyramidal(
                    outputs, sequence_length)

    return (outputs, sequence_length), state


def listener(encoder_inputs,
             source_sequence_length,
             mode,
             num_units,
             keep_prob,
             num_layers):

    return pyramidal_bilstm(encoder_inputs, source_sequence_length, mode,  num_units, keep_prob, num_layers)


def attend(encoder_outputs,
           source_sequence_length,
           mode,
           attention_type,
           attention_size,
           num_units,
           num_layers,
           keep_prob):

    memory = encoder_outputs

    if attention_type == 'luong':
        attention_fn = tf.contrib.seq2seq.LuongAttention
    else:
        attention_fn = tf.contrib.seq2seq.BahdanauAttention

    attention_mechanism = attention_fn(
        num_units, memory, source_sequence_length)

    cell_list = []
    for layer in range(num_layers):
        with tf.variable_scope('decoder_cell_'.format(layer)):
            cell = lstm_cell(size=num_units, activation=None,
                             keep_prob=keep_prob, train_ph=mode == tf.estimator.ModeKeys.TRAIN)

        cell_list.append(cell)

    alignment_history = (mode != tf.estimator.ModeKeys.TRAIN)

    decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=attention_size,
        alignment_history=alignment_history)

    return decoder_cell


def speller(encoder_outputs,
            encoder_state,
            decoder_inputs,
            source_sequence_length,
            target_sequence_length,
            mode,
            beam_width,
            num_embeddings,
            target_vocab_size,
            sampling_probability,
            eos_id, sos_id,
            attention_type,
            attention_size,
            num_attention_units,
            num_attention_layers,
            keep_prob
            ):

    batch_size = tf.shape(encoder_outputs)[0]

    if mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        source_sequence_length = tf.contrib.seq2seq.tile_batch(
            source_sequence_length, multiplier=beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(
            encoder_state, multiplier=beam_width)
        batch_size = batch_size * beam_width

    def embedding_fn(ids):
        # pass callable object to avoid OOM when using one-hot encoding
        if num_embeddings != 0:
            target_embedding = tf.get_variable(
                'target_embedding', [
                    target_vocab_size, num_embeddings],
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            return tf.nn.embedding_lookup(target_embedding, ids)
        else:
            return tf.one_hot(ids, target_vocab_size)

    decoder_cell = attend(
        encoder_outputs, source_sequence_length, mode,
        attention_type=attention_type,
        attention_size=attention_size,
        num_units=num_attention_units,
        num_layers=num_attention_layers,
        keep_prob=keep_prob
    )

    projection_layer = tf.layers.Dense(
        target_vocab_size, use_bias=True, name='projection_layer')

    initial_state = decoder_cell.zero_state(batch_size, tf.float32)

    maximum_iterations = None
    if mode != tf.estimator.ModeKeys.TRAIN:
        max_source_length = tf.reduce_max(source_sequence_length)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(
            max_source_length) * 2))

    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_inputs = embedding_fn(decoder_inputs)

        if sampling_probability > 0.0:
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                decoder_inputs, target_sequence_length,
                embedding_fn, sampling_probability)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs, target_sequence_length)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)

    elif mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
        start_tokens = tf.fill(
            [tf.div(batch_size, beam_width)], sos_id)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding_fn,
            start_tokens=start_tokens,
            end_token=eos_id,
            initial_state=initial_state,
            beam_width=beam_width,
            output_layer=projection_layer)
    else:
        start_tokens = tf.fill([batch_size], sos_id)

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_fn, start_tokens, eos_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, initial_state, output_layer=projection_layer)

    decoder_outputs, final_context_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=maximum_iterations)

    return decoder_outputs, final_context_state, final_sequence_length
