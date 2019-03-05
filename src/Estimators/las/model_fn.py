import tensorflow as tf
from src.neural_network.network_utils import bidirectional_pyramidal_rnn, lstm_cell


def model_fn(features, labels, mode, params):

    global_step = tf.train.get_global_step()

    with tf.name_scope("input_features"):
        input_features = features['feature']

    with tf.name_scope("input_features_length"):
        input_features_length = features['feat_len']

    input_labels = None
    input_labels_length = None
    max_label_length = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("input_labels"):
            input_labels = labels['targets_outputs']

        with tf.name_scope("input_labels_length"):
            input_labels_length = labels['target_len']

        max_label_length = tf.reduce_max(input_labels_length, name='max_label_length')

    max_features_length = tf.reduce_max(input_features_length, name='max_features_length')
    batch_size = tf.shape(input_features)[0]

    with tf.name_scope("listener"):
        listener_output, listener_out_len, listener_state = bidirectional_pyramidal_rnn(
            input_ph=input_features,
            seq_len_ph=input_features_length,
            num_layers=params['listener_num_layers'],
            num_units=params['listener_num_units'],
            name="listener",
            activation_list=params['listener_activation_list'],
            use_tensorboard=True,
            tensorboard_scope="listener",
            keep_prob=params['listener_keep_prob_list'],
            train_ph=tf.constant(mode == tf.estimator.ModeKeys.TRAIN, dtype=tf.bool))

    with tf.name_scope("embeddings"):
        def embedding_fn(ids):
            if params['num_embeddings'] > 0:
                embedding = tf.get_variable(name='embedding',
                                            shape=[params['num_classes'], params['num_embeddings']],
                                            dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())

                return tf.nn.embedding_lookup(params=embedding,
                                              ids=ids,
                                              name='label_embedding')
            else:
                return tf.one_hot(ids, params['num_classes'])

    if mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
        listener_output = tf.contrib.seq2seq.tile_batch(
            listener_output, multiplier=params['beam_width'])
        listener_state = tf.contrib.seq2seq.tile_batch(
            listener_state, multiplier=params['beam_width'])
        listener_out_len = tf.contrib.seq2seq.tile_batch(
            listener_out_len, multiplier=params['beam_width'])
        batch_size = batch_size * params['beam_width']

    with tf.variable_scope("attention"):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(          # LuongAttention(
            num_units=params['attention_units'],
            memory=listener_output,
            memory_sequence_length=listener_out_len)

        attention_cell_list = []
        for layer in range(params['attention_num_layers']):
            with tf.variable_scope('decoder_cell_'.format(layer)):
                cell = lstm_cell(
                    size=params['attention_rnn_units'][layer],
                    activation=params['attention_activation_list'][layer],
                    keep_prob=params['attention_keep_prob_list'][layer],
                    train_ph=tf.constant(mode == tf.estimator.ModeKeys.TRAIN, dtype=tf.bool))

            attention_cell_list.append(cell)

        attention_cell = tf.nn.rnn_cell.MultiRNNCell(attention_cell_list)

        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
            attention_cell, attention_mechanism,
            attention_layer_size=None,
            alignment_history=mode != tf.estimator.ModeKeys.TRAIN)

        projection_layer = tf.layers.Dense(
            params['num_classes'], use_bias=True, name='projection_layer')

        initial_state = attention_cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("decoding"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            if params['sampling_probability'] > 0.0:
                helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    embedding_fn(labels['targets_inputs']), input_labels_length,
                    embedding_fn, params['sampling_probability'])
            else:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    input_labels, input_labels_length)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                attention_cell, helper, initial_state, output_layer=projection_layer)

        elif mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            start_tokens = tf.fill(
                [tf.div(batch_size, params['beam_width'])], params['sos_id'])

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=attention_cell,
                embedding=embedding_fn,
                start_tokens=start_tokens,
                end_token=params['eos_id'],
                initial_state=initial_state,
                beam_width=params['beam_width'],
                output_layer=projection_layer)

        else:
            start_tokens = tf.fill([batch_size], params['sos_id'])

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding_fn, start_tokens, params['eos_id'])

            decoder = tf.contrib.seq2seq.BasicDecoder(
                attention_cell, helper, initial_state, output_layer=projection_layer)

        maximum_iterations = None
        if mode != tf.estimator.ModeKeys.TRAIN:
            maximum_iterations = max_features_length * 2

        decoder_outputs, final_context_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)

    with tf.name_scope('prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            sample_ids = tf.to_int32(tf.argmax(logits, -1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        # predictions = {
        #     'sample_ids': sample_ids,
        # }
        return tf.estimator.EstimatorSpec(mode, predictions=sample_ids)

    with tf.name_scope('metrics'):
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

            hypothesis = dense_to_sparse(hypothesis, eos_id, merge_repeated=True)
            truth = dense_to_sparse(truth, eos_id, merge_repeated=True)

            return tf.edit_distance(hypothesis, truth, normalize=True)
        ler = edit_distance(sample_ids, input_labels, params['eos_id'])
        # tf.reduce_mean(tf.edit_distance(
        #     hypothesis=tf.contrib.layers.dense_to_sparse(tf.cast(sample_ids, tf.int32)),
        #     truth=tf.contrib.layers.dense_to_sparse(input_labels),
        #     normalize=True))
        metrics = {
            'edit_distance': tf.metrics.mean(ler),
        }

        tf.summary.scalar('edit_distance', metrics['edit_distance'][1])

    with tf.name_scope("loss"):
        kernel_loss = 0
        for var in tf.trainable_variables():
            if var.name.startswith('dense_layer') and 'kernel' in var.name:
                kernel_loss += tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if var.name.startswith('listener') and 'kernel' in var.name:
                kernel_loss += tf.nn.l2_loss(var)

        for var in tf.trainable_variables():
            if var.name.startswith('attention') and 'kernel' in var.name:
                kernel_loss += tf.nn.l2_loss(var)

        target_weights = tf.sequence_mask(input_labels_length, dtype=tf.float32, name='mask')

        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=input_labels,
                                                         weights=target_weights)

        loss = sequence_loss + params['kernel_regularizer'] * kernel_loss
        tf.summary.scalar('sequence_loss', sequence_loss)
        tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        # with tf.name_scope('alignment'):
        #     attention_images = utils.create_attention_images(
        #         final_context_state)

        # attention_summary = tf.summary.image(
        #     'attention_images', attention_images)

        logging_hook = tf.train.LoggingTensorHook({
            'edit_distance': tf.reduce_mean(ler),
        }, every_n_iter=10)

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[logging_hook])

    with tf.name_scope('train'):
        optimizer = params['optimizer']
        train_op = optimizer.minimize(
            loss, global_step=global_step)

    logging_hook = tf.train.LoggingTensorHook({
        'loss': loss,
        'ler': tf.reduce_mean(ler),
    }, every_n_secs=1)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])



