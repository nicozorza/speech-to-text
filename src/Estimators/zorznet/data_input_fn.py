import tensorflow as tf


def data_input_fn(filenames, batch_size, parse_fn, shuffle_buffer, num_features, num_epochs=1):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_fn)
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=((None, num_features), [None], (), ()),
        padding_values=(tf.constant(value=0, dtype=tf.float32),
                        tf.constant(value=-1, dtype=tf.int64),
                        tf.constant(value=0, dtype=tf.int64),
                        tf.constant(value=0, dtype=tf.int64),
                        )
    )
    dataset = dataset.shuffle(shuffle_buffer).repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    feature, target, feat_len, target_len = next_element

    feat_len = tf.cast(feat_len, dtype=tf.int32)
    target = tf.cast(target, tf.int32)
    sparse_target = tf.contrib.layers.dense_to_sparse(target, eos_token=-1)

    return {'feature': feature, 'feat_len': feat_len}, sparse_target
