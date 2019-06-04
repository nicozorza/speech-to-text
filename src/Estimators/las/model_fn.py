import os

import tensorflow as tf
from src.neural_network.network_utils import bidirectional_pyramidal_rnn, attention_layer, attention_decoder, \
    beam_search_decoder, greedy_decoder, edit_distance, attention_loss, dense_multilayer


def model_fn(features,
             labels,
             mode,
             config,
             params):

    input_features = features['feature']
    input_features_length = features['feat_len']

    if params['noise_stddev'] is not None and params['noise_stddev'] != 0.0:
        input_features = tf.keras.layers.GaussianNoise(stddev=params['noise_stddev'])(inputs=input_features, training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder_inputs = None
    targets = None
    targets_length = None

    global_step = tf.train.get_global_step()

    if mode != tf.estimator.ModeKeys.PREDICT:
        decoder_inputs = labels['targets_inputs']
        targets = labels['targets_outputs']
        targets_length = labels['target_len']

    with tf.name_scope("dense_layer_1"):
        input_features = dense_multilayer(input_ph=input_features,
                                          num_layers=params['num_dense_layers_1'],
                                          num_units=params['num_units_1'],
                                          name='dense_layer_1',
                                          activation_list=params['dense_activations_1'],
                                          use_batch_normalization=params['batch_normalization_1'],
                                          batch_normalization_trainable=params['batch_normalization_trainable_1'],
                                          train_ph=mode == tf.estimator.ModeKeys.TRAIN,
                                          use_tensorboard=True,
                                          keep_prob_list=params['keep_prob_1'],
                                          kernel_initializers=params['kernel_init_1'],
                                          bias_initializers=params['bias_init_1'],
                                          tensorboard_scope='dense_layer_1')

    with tf.variable_scope('listener'):
        listener_output, input_features_length, listener_state = bidirectional_pyramidal_rnn(
            input_ph=input_features,
            seq_len_ph=input_features_length,
            num_layers=params['listener_num_layers'],
            num_units=params['listener_num_units'],
            name="listener",
            activation_list=params['listener_activation_list'],
            use_tensorboard=True,
            tensorboard_scope="listener",
            keep_prob=params['listener_keep_prob_list'],
            train_ph=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope("dense_layer_2"):
        listener_output = dense_multilayer(input_ph=listener_output,
                                           num_layers=params['num_dense_layers_2'],
                                           num_units=params['num_units_2'],
                                           name='dense_layer_2',
                                           activation_list=params['dense_activations_2'],
                                           use_batch_normalization=params['batch_normalization_2'],
                                           batch_normalization_trainable=params['batch_normalization_trainable_2'],
                                           train_ph=mode == tf.estimator.ModeKeys.TRAIN,
                                           use_tensorboard=True,
                                           keep_prob_list=params['keep_prob_2'],
                                           kernel_initializers=params['kernel_init_2'],
                                           bias_initializers=params['bias_init_2'],
                                           tensorboard_scope='dense_layer_2')

    with tf.variable_scope('tile_batch'):
        batch_size = tf.shape(listener_output)[0]
        if mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            listener_output = tf.contrib.seq2seq.tile_batch(
                listener_output, multiplier=params['beam_width'])
            input_features_length = tf.contrib.seq2seq.tile_batch(
                input_features_length, multiplier=params['beam_width'])
            listener_state = tf.contrib.seq2seq.tile_batch(
                listener_state, multiplier=params['beam_width'])
            batch_size = batch_size * params['beam_width']

    with tf.variable_scope('attention'):
        attention_cell, attention_state = attention_layer(
            input=listener_output,
            lengths=input_features_length,
            num_layers=params['attention_num_layers'],
            attention_units=params['attention_units'],
            attention_size=params['attention_size'],
            attention_type=params['attention_type'],
            activation=params['attention_activation'],
            keep_prob=params['attention_keep_prob'],
            train_ph=mode == tf.estimator.ModeKeys.TRAIN,
            batch_size=batch_size,
            input_state=None,
            use_tensorboard=True,
            tensorboard_scope='attention_cell'
        )

    with tf.variable_scope('speller'):
        def embedding_fn(ids):
            if params['num_embeddings'] != 0:
                target_embedding = tf.get_variable(
                    name='target_embedding',
                    shape=[params['num_classes'], params['num_embeddings']],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                return tf.nn.embedding_lookup(target_embedding, ids)
            else:
                return tf.one_hot(ids, params['num_classes'])

        projection_layer = tf.layers.Dense(params['num_classes'], use_bias=True, name='projection_layer')

        maximum_iterations = None
        if mode != tf.estimator.ModeKeys.TRAIN:
            max_source_length = tf.reduce_max(input_features_length)
            maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_source_length) * 2))

        if mode == tf.estimator.ModeKeys.TRAIN:
            decoder_inputs = embedding_fn(decoder_inputs)

            decoder = attention_decoder(
                input_cell=attention_cell,
                initial_state=attention_state,
                embedding_fn=embedding_fn,
                seq_embedding=decoder_inputs,
                seq_embedding_len=targets_length,
                projection_layer=projection_layer,
                sampling_prob=params['sampling_probability'])

        elif mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            decoder = beam_search_decoder(
                input_cell=attention_cell,
                embedding=embedding_fn,
                start_token=params['sos_id'],
                end_token=params['eos_id'],
                initial_state=attention_state,
                beam_width=params['beam_width'],
                projection_layer=projection_layer,
                batch_size=batch_size)
        else:

            decoder = greedy_decoder(
                inputs=attention_cell,
                embedding=embedding_fn,
                start_token=params['sos_id'],
                end_token=params['eos_id'],
                initial_state=attention_state,
                projection_layer=projection_layer,
                batch_size=batch_size)

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
        predictions = {'sample_ids': sample_ids}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope('metrics'):
        ler = edit_distance(
            sample_ids, targets, params['eos_id'], None) #params.mapping)

        metrics = {'LER': tf.metrics.mean(ler),}

    tf.summary.scalar('LER', metrics['LER'][1])

    with tf.name_scope('loss'):
        kernel_loss = 0
        for var in tf.trainable_variables():
            if var.name.startswith('dense_layer') and 'kernel' in var.name:
                kernel_loss += tf.nn.l2_loss(var)

        attetion_loss = attention_loss(
            logits=logits,
            targets=targets,
            logits_length=final_sequence_length,
            targets_length=targets_length,
            eos_id=params['eos_id'],
            train_ph=mode == tf.estimator.ModeKeys.TRAIN)

        loss = attetion_loss + params['kernel_regularizer'] * kernel_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        def _create_attention_images_summary(context_state):
            """Reference: https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py"""
            images = (context_state.alignment_history.stack())
            # Reshape to (batch, src_seq_len, tgt_seq_len,1)
            images = tf.expand_dims(tf.transpose(images, [1, 2, 0]), -1)
            # Scale to range [0, 255]
            images -= 1
            images = -images
            images *= 255
            summary = tf.summary.image("attention_images", images)
            return summary
        with tf.name_scope('alignment'):
            attention_summary = _create_attention_images_summary(final_context_state)

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            output_dir=os.path.join(config.model_dir, 'eval'),
            summary_op=attention_summary)

        logging_hook = tf.train.LoggingTensorHook(
            tensors={
                'LER': tf.reduce_mean(ler),
                # 'max_predictions': sample_ids[tf.argmax(ler)],
                # 'max_targets': targets[tf.argmax(ler)],
            },
            every_n_iter=10)

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics,
                                          evaluation_hooks=[logging_hook, eval_summary_hook])

    with tf.name_scope('train'):
        if params['use_learning_rate_decay']:
            learning_rate = tf.train.exponential_decay(
                params['learning_rate'],
                global_step,
                decay_steps=params['learning_rate_decay_steps'],
                decay_rate=params['learning_rate_decay'],
                staircase=True)
        else:
            learning_rate = params['learning_rate']

        if params['optimizer'] == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif params['optimizer'] == 'momentum' and params['momentum'] is not None:
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params['momentum'])
        elif params['optimizer'] == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if params['clip_gradient'] != 0:
                grads = tf.gradients(loss, tf.trainable_variables())
                grads, _ = tf.clip_by_global_norm(grads, params['clip_gradient'])  # gradient clipping
                grads_and_vars = list(zip(grads, tf.trainable_variables()))
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            else:
                train_op = optimizer.minimize(loss, global_step=global_step)

    logging_hook = tf.train.LoggingTensorHook(
        tensors={
            'loss': loss,
            'LER': tf.reduce_mean(ler),
            'learning_rate': tf.reduce_mean(learning_rate)
        },
        every_n_secs=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                      training_hooks=[logging_hook], eval_metric_ops=metrics)



