# -*- coding: utf-8 -*-

""" Train a ML model

    TODO:

    *   Parameterize Optimizer and learning rate
"""

import logging
import utils
import models

import argparse
import datetime
import math
import os
import json
import gc
import sys

import tensorflow as tf
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform import TFTransformOutput
from tensorflow.python.lib.io import file_io

from tensorflow.keras import backend as K

import tensorflow_model_analysis as tfma

import multiprocessing
N_CORES = multiprocessing.cpu_count()

tf.logging.set_verbosity(tf.logging.INFO)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def input_fn(tfrecords_path,
             tft_metadata,
             window_size,
             batch_size=8):
    """ Train input function

        Create and parse dataset from tfrecords shards with TFT schema
    """

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

    num_parallel_calls = N_CORES-1
    if num_parallel_calls <= 0:
        num_parallel_calls = 1

    dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type="")
    dataset = dataset.map(_parse_sequence_example,
                          num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(_split_XY, num_parallel_calls=num_parallel_calls)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat()

    return dataset


def get_raw_feature_spec(window_size):
    """ Retrieve schema for input features to be used at serving function
        
        Unfortunately TFT does not dump input feature spec, so it
        needs to be hard-coded built
    """

    feature_spec = {
        'hour': tf.FixedLenFeature(shape=[window_size], dtype=tf.int64, default_value=None),
        'day_of_week': tf.FixedLenFeature(shape=[window_size], dtype=tf.int64, default_value=None),
        'day_of_month': tf.FixedLenFeature(shape=[window_size], dtype=tf.int64, default_value=None),
        'week_number': tf.FixedLenFeature(shape=[window_size], dtype=tf.int64, default_value=None),
        'month': tf.FixedLenFeature(shape=[window_size], dtype=tf.int64, default_value=None),
        'community_area': tf.FixedLenFeature(shape=[window_size], dtype=tf.int64, default_value=None),
        'n_trips': tf.FixedLenFeature(shape=[window_size], dtype=tf.float32, default_value=None),
        'community_area_code': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=None),
    }

    return feature_spec


def serving_input_receiver_fn(tft_metadata, window_size):
    """ Return serving function for model deployment
    """

    raw_feature_spec = get_raw_feature_spec(window_size)

    def _serving_input_receiver_fn():

        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            raw_feature_spec, default_batch_size=None)
        serving_input_receiver = raw_input_fn()

        # In order to work on CMLE, it is expected the input tensor to be encoded as base64
        decoded_example = tf.io.decode_base64(
            serving_input_receiver.receiver_tensors['examples'])
        raw_features = tf.io.parse_example(decoded_example, raw_feature_spec)

        transformed_features = tft_metadata.transform_raw_features(
            raw_features)

        # remove target tensor
        transformed_features.pop('target')

        return tf.estimator.export.ServingInputReceiver(
            transformed_features, serving_input_receiver.receiver_tensors)

    return _serving_input_receiver_fn


def eval_input_receiver_fn(tft_metadata, window_size):
    """ Return serving function for model analysis
    """

    raw_feature_spec = get_raw_feature_spec(window_size)

    # Add target to raw feature spec
    raw_feature_spec['target'] = tf.FixedLenFeature(
        shape=[], dtype=tf.float32, default_value=None)

    input_proto = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_placeholder')

    features = tf.io.parse_example(input_proto, raw_feature_spec)

    transformed_features = tft_metadata.transform_raw_features(
        features)
    target_tensor = {'target': transformed_features['target']}

    # Remove target tensor from transformed feature,
    # once transformed_features will be input to the model
    transformed_features.pop('target')

    receiver_tensors = {'examples': input_proto}

    return tfma.export.EvalInputReceiver(
        features=transformed_features,
        receiver_tensors=receiver_tensors,
        labels=target_tensor)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--tfrecord-file-train',
                        dest='tfrecord_file_train', required=True)
    parser.add_argument('--tfrecord-file-eval',
                        dest='tfrecord_file_eval', required=True)
    parser.add_argument('--tft-artifacts-dir',
                        dest='tft_artifacts_dir', required=True)

    parser.add_argument('--n-windows-train',
                        dest='n_windows_train', required=True, type=int)
    parser.add_argument('--n-windows-eval',
                        dest='n_windows_eval', required=True, type=int)
    parser.add_argument('--window-size',
                        dest='window_size', required=True, type=int)

    parser.add_argument('--model-name', dest='model_name',
                        required=True, type=str)
    parser.add_argument('--n-areas', dest='n_areas',
                        required=True, type=int)

    parser.add_argument('--epochs', dest='epochs',
                        required=False, type=int, default=1)
    parser.add_argument('--batch-size', dest='batch_size',
                        required=False, type=int, default=8)

    parser.add_argument('--output-dir', dest='output_dir',
                        required=True, type=str)
    parser.add_argument('--job-dir', dest='job_dir',
                        required=False, type=str, default='/tmp/')
    parser.add_argument('--gpu-memory-fraction', dest='gpu_memory_fraction',
                        required=False, type=float, default=0.8)

    args = parser.parse_args()

    logger.info("args:\n{}".format(args))

    steps_per_epoch_train = int(
        math.ceil(args.n_windows_train / args.batch_size))
    logger.info('steps_per_epoch_train: {}'.format(steps_per_epoch_train))

    steps_per_epoch_eval = int(
        math.ceil(args.n_windows_eval / args.batch_size))
    logger.info('steps_per_epoch_eval: {}'.format(steps_per_epoch_eval))

    logger.info('Reading TFT artifacts')
    tft_metadata_dir = os.path.join(
        args.tft_artifacts_dir, transform_fn_io.TRANSFORM_FN_DIR)
    tft_metadata = TFTransformOutput(args.tft_artifacts_dir)
    logger.info('Done')

    logger.info('Listing tfrecords ...')
    tfrecords_train_list = utils.list_tfrecords(args.tfrecord_file_train)
    tfrecords_eval_list = utils.list_tfrecords(args.tfrecord_file_eval)
    logger.info("{} train tfrecords ".format(len(tfrecords_train_list)))
    logger.info("{} eval tfrecords ".format(len(tfrecords_eval_list)))
    logger.info('Done!')

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        logger.info('Loading {} model'.format(args.model_name))
        model = models.get_model(
            args.model_name, window_size=args.window_size, n_areas=args.n_areas)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=5e-4),
                      loss='mse',
                      metrics=['mse'])
        logger.info(model.summary())
        logger.info('Done!')

        logger.info('Loading input_fn')

        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tfrecords_train_list,
                                      tft_metadata,
                                      args.window_size,
                                      args.batch_size),
            max_steps=steps_per_epoch_train*args.epochs)

        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(tfrecords_eval_list,
                                      tft_metadata,
                                      args.window_size,
                                      args.batch_size),
            steps=steps_per_epoch_eval*args.epochs)
        # start_delay_secs=60,
        # throttle_secs=30)
        logger.info('Done')

        datetime_now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_dir = os.path.join(args.output_dir, datetime_now_str)
        logger.info('Dumping artifacts at: {}'.format(output_dir))

        run_config = tf.estimator.RunConfig(
            model_dir=output_dir,
            save_summary_steps=1000,
            save_checkpoints_steps=1000,
            keep_checkpoint_max=1
        )

        # Convert keras to estimator
        model_estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
                                                                config=run_config)

        logger.info('Train')
        tf.estimator.train_and_evaluate(estimator=model_estimator,
                                        train_spec=train_spec,
                                        eval_spec=eval_spec)

        # export saved model
        export_dir = os.path.join(output_dir, 'export')
        if tf.gfile.Exists(export_dir):
            tf.gfile.DeleteRecursively(export_dir)

        export_saved_model_path = model_estimator.export_savedmodel(
            export_dir_base=export_dir,
            serving_input_receiver_fn=serving_input_receiver_fn(tft_metadata,
                                                                args.window_size)
        )

        # export eval saved model for TensorFlow Model Analysis
        eval_model_dir = os.path.join(output_dir, 'eval_saved_model')
        if tf.gfile.Exists(eval_model_dir):
            tf.gfile.DeleteRecursively(eval_model_dir)

        export_eval_saved_model_path = tfma.export.export_eval_savedmodel(
            estimator=model_estimator,
            export_dir_base=eval_model_dir,
            eval_input_receiver_fn=lambda: eval_input_receiver_fn(tft_metadata, args.window_size))

        # Export output_dir to access Tensorboard through Kubeflow UI
        metadata = {
            'outputs': [{
                'type': 'tensorboard',
                'source': output_dir,
            }]
        }
        with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
            json.dump(metadata, f)

        logger.info('Saved model exported at: {}'.format(
            export_saved_model_path))
        with file_io.FileIO('/saved_model_path.txt', 'w') as f:
            f.write(export_saved_model_path)

        logger.info('Eval saved model exported at {}'.format(
            export_eval_saved_model_path))
        with file_io.FileIO('/eval_saved_model_path.txt', 'w') as f:
            f.write(export_eval_saved_model_path)

        logger.info("Training Done")
    gc.collect()  # https://github.com/tensorflow/tensorflow/issues/3388#issuecomment-268502675
    K.clear_session()
    sys.exit(0)
