import os
import tensorflow as tf
from src.neural_network.network_utils import dense_multilayer, dense_layer, multihead_attention, \
    add_positional_encoding


def model_fn(features, labels, mode, config, params):

    feature = features['feature']
    feat_len = features['feat_len']
    sparse_target = labels

    global_step = tf.train.get_global_step()

    with tf.name_scope("seq_len"):
        input_features_length = feat_len

    with tf.name_scope("input_features"):
        input_features = feature

    with tf.name_scope("input_labels"):
        input_labels = sparse_target

    subsample_factor = params["num_reduce_by_half"]
    if subsample_factor is not None and subsample_factor > 0:
        for i in range(subsample_factor):
            input_features_length = tf.div(input_features_length, 2) + tf.cast(input_features_length % 2, dtype=tf.int32)
            input_features = input_features[:, ::2]

    if params['noise_stddev'] is not None and params['noise_stddev'] != 0.0:
        input_features = tf.keras.layers.GaussianNoise(stddev=params['noise_stddev'])(inputs=input_features, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense_1_input = tf.identity(input_features)
    with tf.name_scope("dense_layer_1"):
        dense_1_output = dense_multilayer(input_ph=dense_1_input,
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

    with tf.name_scope("attention"):
        if params["attention_add_positional_encoding"]:
            dense_1_output = add_positional_encoding(dense_1_output, encoding_type=params["positional_encoding_type"])

        attention_output = multihead_attention(
            input_ph=dense_1_output,
            input_len=input_features_length,
            num_heads=params['attention_num_heads'],
            hidden_dim=params['attention_hidden_size'],
            hidden_output=params['attention_hidden_output_size'],
            output_dim=params['attention_output_size'],
            activation=params['attention_activation'],
            scaled=params["attention_scaled"],
            masked=params["attention_masked"]
        )
        if params["attention_use_layer_normalization"]:
            attention_output = tf.contrib.layers.layer_norm(
                attention_output,
                trainable=params["attention_layer_normalization_trainable"])

    with tf.name_scope("dense_layer_2"):
        dense_2_output = dense_multilayer(input_ph=attention_output,
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

    with tf.name_scope("dense_output"):
        dense_output_no_activation = dense_layer(input_ph=dense_2_output,
                                                 num_units=params['num_classes'],
                                                 name='dense_output_no_activation',
                                                 activation=None,
                                                 use_batch_normalization=False,
                                                 train_ph=False,
                                                 use_tensorboard=True,
                                                 keep_prob=1,
                                                 tensorboard_scope='dense_output')

        dense_output = tf.nn.softmax(dense_output_no_activation, name='dense_output')
        tf.summary.histogram('dense_output', dense_output)

    with tf.name_scope("decoder"):
        output_time_major = tf.transpose(dense_output, (1, 0, 2))
        if params['beam_width'] == 0:
            decoded, log_prob = tf.nn.ctc_greedy_decoder(output_time_major, input_features_length, merge_repeated=True)
        else:
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(output_time_major, input_features_length,
                                                              beam_width=params['beam_width'],
                                                              top_paths=1,
                                                              merge_repeated=False)
        dense_decoded = tf.sparse.to_dense(sp_input=decoded[0], validate_indices=True)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=dense_decoded)

    with tf.name_scope("loss"):
        dense_loss = 0
        for var in tf.trainable_variables():
            if var.name.startswith('dense_layer') or \
                    var.name.startswith('input_dense_layer') and \
                    'kernel' in var.name:
                dense_loss += tf.nn.l2_loss(var)

        attention_loss = 0
        for var in tf.trainable_variables():
            if var.name.startswith('head_') and 'kernel' in var.name:
                attention_loss += tf.nn.l2_loss(var)

        loss = tf.nn.ctc_loss(input_labels, dense_output_no_activation, input_features_length,
                              time_major=False)
        logits_loss = tf.reduce_mean(tf.reduce_sum(loss))
        loss = logits_loss \
               + params['dense_regularizer'] * dense_loss \
               + params["attention_regularizer"] * attention_loss
        tf.summary.scalar('loss', loss)

    with tf.name_scope("label_error_rate"):
        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(decoded[0], tf.int32),
                                              truth=input_labels,
                                              normalize=True))
        metrics = {'ler': tf.metrics.mean(ler), }
        tf.summary.scalar('label_error_rate', tf.reduce_mean(ler))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={
            "loss": loss,
            "ler": ler,
        },
        every_n_iter=1
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
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

        loss = tf.tuple([loss], control_inputs=tf.get_collection(tf.GraphKeys.UPDATE_OPS))[0]
        if params['clip_gradient'] != 0:
            grads = tf.gradients(loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, params['clip_gradient'])
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

        train_logging_hook = tf.train.LoggingTensorHook(
            tensors={
                'loss': loss,
                'ler': tf.reduce_mean(ler),
                'learning_rate': tf.reduce_mean(learning_rate),
                # 'feal_len': feat_len,
                # 'feal_len2': input_features_length,
                # 'feal_len3': tf.shape(input_features),
                # 'target_len': tf.shape(input_labels)
                # 'max_predictions': dense_decoded,
                # 'max_targets': tf.sparse.to_dense(sp_input=input_labels, validate_indices=True),
            },
            every_n_secs=1)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            training_hooks=[train_logging_hook],
            eval_metric_ops=metrics
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        def _create_alignment_images_summary(outputs):
            images = outputs
            images = tf.expand_dims(images, -1)
            # Scale to range [0, 255]
            images -= 1
            images = -images
            images *= 255
            summary = tf.summary.image("alignment_images", images)
            return summary
        with tf.name_scope('alignment'):
            alignment_summary = _create_alignment_images_summary(dense_output)

        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=10,
            output_dir=os.path.join(config.model_dir, 'eval'),
            summary_op=alignment_summary)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            evaluation_hooks=[logging_hook, eval_summary_hook],
            eval_metric_ops=metrics
        )


