# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
import numpy as np
import argparse
from tensorflow_transform import TFTransformOutput
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
import tensorflow as tf
import utils
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("agg")


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def input_fn(tfrecords_path,
             tft_metadata,
             batch_size=8):

    def _parse_sequence_example(proto):
        return tf.io.parse_single_example(proto, features=tft_metadata.transformed_feature_spec())

    def _split_XY(example):
        X = {}
        Y = {}

        for k in example.keys():
            if k != 'target':
                X[k] = example[k]
        Y['target'] = example['target']

        return X, Y

    dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type="")
    dataset = dataset.map(_parse_sequence_example)
    dataset = dataset.map(_split_XY)
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--tfrecord-file',
                        dest='tfrecord_file', required=True)
    parser.add_argument('--tft-artifacts-dir',
                        dest='tft_artifacts_dir', required=True)
    args = parser.parse_args()

    tfrecords_list = utils.list_tfrecords(args.tfrecord_file)

    tft_metadata_dir = os.path.join(
        args.tft_artifacts_dir, transform_fn_io.TRANSFORM_FN_DIR)
    tft_metadata = TFTransformOutput(args.tft_artifacts_dir)

    input_fn_op = input_fn(tfrecords_list, tft_metadata, 1)

    input_fn_next = input_fn_op.make_one_shot_iterator().get_next()

    data_dict = {
        'community_area_code': [],
        'target': []
    }

    stop = False
    with tf.Session() as sess:
        while True:
            try:
                batch_X, batch_Y = sess.run(input_fn_next)
                for k in batch_X.keys():
                    if np.any(np.isnan(batch_X[k])):
                        logger.info("{} {}".format(k, batch_X[k]))
                        stop = True
                        break                
                data_dict['community_area_code'].append(
                    batch_X['community_area_code'][0])
                data_dict['target'].append(batch_Y['target'][0])
            except tf.errors.OutOfRangeError:
                break

            if stop:
                break
    data_df = pd.DataFrame(data_dict)
    
    os.makedirs(os.path.join(args.tft_artifacts_dir,'images'),exist_ok=True)


    for ca, ca_group in data_df.groupby('community_area_code'):
        plt.figure(figsize=(10, 3))
        plt.title('Target - {}'.format(ca))
        plt.plot(ca_group['target'].values)
        plt.savefig(os.path.join(args.tft_artifacts_dir,
                                 'images', '{}.png'.format(ca)))
        plt.close()
