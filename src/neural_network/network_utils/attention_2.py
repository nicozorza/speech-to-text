import tensorflow as tf
from src.neural_network.network_utils import lstm_cell


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


def dense_to_sparse(tensor, eos_id, merge_repeated=True):
    if merge_repeated:
        added_values = tf.cast(
            tf.fill((tf.shape(tensor)[0], 1), eos_id), tensor.dtype)

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


def compute_loss(logits, targets, final_sequence_length, target_sequence_length, mode):

    assert mode != tf.estimator.ModeKeys.PREDICT

    target_weights = tf.sequence_mask(
        target_sequence_length, dtype=tf.float32)
    loss = tf.contrib.seq2seq.sequence_loss(
        logits, targets, target_weights)

    return loss