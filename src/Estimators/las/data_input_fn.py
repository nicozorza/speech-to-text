import tensorflow as tf


def data_input_fn(filenames, batch_size, parse_fn, shuffle_buffer, num_features, sos_id, eos_id, num_epochs=1):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_fn)

    dataset = dataset.map(
        lambda feature, target, feat_len, target_len: (feature,
                                                       tf.concat(([sos_id], target), 0),
                                                       tf.concat((target, [eos_id]), 0),
                                                       feat_len,
                                                       target_len)
    )

    dataset = dataset.map(
        lambda feature, target_in, target_out, feat_len, target_len: (feature,
                                                                      target_in,
                                                                      target_out,
                                                                      feat_len,
                                                                      tf.size(target_in, out_type=tf.int64)))

    # dataset = dataset.map(
    #     lambda feature, target_in, target_out, feat_len, target_len: (feature,
    #                                                                   tf.cast(target_in, tf.int32),
    #                                                                   tf.cast(target_out, tf.int32),
    #                                                                   tf.cast(feat_len, dtype=tf.int32),
    #                                                                   tf.cast(target_len, dtype=tf.int32)
    #                                                                   )
    # )

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=((None, num_features), [None], [None], (), ()),
        padding_values=(tf.constant(value=0, dtype=tf.float32),
                        tf.constant(value=eos_id, dtype=tf.int64),
                        tf.constant(value=eos_id, dtype=tf.int64),
                        tf.constant(value=0, dtype=tf.int64),
                        tf.constant(value=0, dtype=tf.int64),
                        )
    )
    dataset = dataset.shuffle(shuffle_buffer).repeat(num_epochs)

    # return dataset
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    feature, target_in, target_out, feat_len, target_len = next_element

    feat_len = tf.cast(feat_len, dtype=tf.int32)
    target_in = tf.cast(target_in, tf.int32)
    target_out = tf.cast(target_out, tf.int32)
    target_len = tf.cast(target_len, tf.int32)

    return {'feature': feature, 'feat_len': feat_len}, \
           {'targets_outputs': target_out, 'targets_inputs': target_in, 'target_len': target_len}
