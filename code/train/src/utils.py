# -*- coding: utf-8 -*-

import tensorflow as tf

def list_tfrecords(path_regex):
    tfrecord_list = []
    with tf.Session() as sess:
        tfrecord_file_op = tf.data.Dataset.list_files(
            path_regex).make_one_shot_iterator().get_next()
        while True:
            try:
                next_filename = sess.run(tfrecord_file_op)
                tfrecord_list.append(next_filename.decode())
            except tf.errors.OutOfRangeError as err:
                break

    return tfrecord_list
