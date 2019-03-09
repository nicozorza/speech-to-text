import tensorflow as tf
from src.neural_network import network_utils as net_utils
from src.neural_network.network_utils import bidirectional_pyramidal_rnn
from src.neural_network.network_utils.attention_2 import speller, listener


def las_model_fn(features,
                 labels,
                 mode,
                 params):

    encoder_inputs = features['feature']
    source_sequence_length = features['feat_len']

    decoder_inputs = None
    targets = None
    target_sequence_length = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        decoder_inputs = labels['targets_inputs']
        targets = labels['targets_outputs']
        target_sequence_length = labels['target_len']

    tf.logging.info('Building listener')

    with tf.variable_scope('listener'):
        # (encoder_outputs, source_sequence_length), encoder_state = listener(encoder_inputs,
        #          source_sequence_length,
        #          mode,
        #          params['listener_num_units'][0],
        #          params['listener_keep_prob_list'][0],
        #          params['listener_num_layers'])
        encoder_outputs, source_sequence_length, encoder_state = bidirectional_pyramidal_rnn(
            input_ph=encoder_inputs,
            seq_len_ph=source_sequence_length,
            num_layers=params['listener_num_layers'],
            num_units=params['listener_num_units'],
            name="listener",
            activation_list=params['listener_activation_list'],
            use_tensorboard=True,
            tensorboard_scope="listener",
            keep_prob=params['listener_keep_prob_list'],
            train_ph=mode == tf.estimator.ModeKeys.TRAIN)

    tf.logging.info('Building speller')

    with tf.variable_scope('speller'):
        decoder_outputs, final_context_state, final_sequence_length = net_utils.attention_2.speller(
            encoder_outputs, encoder_state, decoder_inputs,
            source_sequence_length, target_sequence_length,
            mode,
            beam_width=params['beam_width'],
            num_embeddings=params['num_embeddings'],
            target_vocab_size=params['num_classes'],
            sampling_probability=params['sampling_probability'],
            eos_id=params['eos_id'], sos_id=params['sos_id'],
            attention_type='luong',
            attention_size=params['attention_units'],
            num_attention_units=params['attention_rnn_units'][0],
            num_attention_layers=params['attention_num_layers'],
            keep_prob=0.9)

    with tf.name_scope('prediction'):
        if mode == tf.estimator.ModeKeys.PREDICT and params['beam_width'] > 0:
            logits = tf.no_op()
            sample_ids = decoder_outputs.predicted_ids
        else:
            logits = decoder_outputs.rnn_output
            sample_ids = tf.to_int32(tf.argmax(logits, -1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'sample_ids': sample_ids,
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope('metrics'):
        edit_distance = net_utils.attention.edit_distance(
            sample_ids, targets, params['eos_id'], None) #params.mapping)

        metrics = {
            'edit_distance': tf.metrics.mean(edit_distance),
        }

    tf.summary.scalar('edit_distance', metrics['edit_distance'][1])

    with tf.name_scope('cross_entropy'):
        loss = net_utils.attention.attention_loss(
            logits, targets, final_sequence_length, target_sequence_length, params['eos_id'],
            mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.EVAL:
        # with tf.name_scope('alignment'):
        #     attention_images = utils.create_attention_images(
        #         final_context_state)

        # attention_summary = tf.summary.image(
        #     'attention_images', attention_images)

        # eval_summary_hook = tf.train.SummarySaverHook(
        #     save_steps=10,
        #     output_dir=os.path.join(config.model_dir, 'eval'),
        #     summary_op=attention_summary)

        logging_hook = tf.train.LoggingTensorHook({
            'edit_distance': tf.reduce_mean(edit_distance),
            'max_edit_distance': tf.reduce_max(edit_distance),
            'max_predictions': sample_ids[tf.argmax(edit_distance)],
            'max_targets': targets[tf.argmax(edit_distance)],
            'min_edit_distance': tf.reduce_min(edit_distance),
            'min_predictions': sample_ids[tf.argmin(edit_distance)],
            'min_targets': targets[tf.argmin(edit_distance)],
        }, every_n_iter=10)

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[logging_hook])

    with tf.name_scope('train'):
        optimizer = params['optimizer']
        # optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook({
        'loss': loss,
        'edit_distance': tf.reduce_mean(edit_distance),
        #'max_edit_distance': tf.reduce_max(edit_distance),
        #'predictions': sample_ids[tf.argmax(edit_distance)],
        #'targets': targets[tf.argmax(edit_distance)],
    }, every_n_secs=10)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])



