# -*- coding: utf-8 -*-

import utils
import models

import argparse
import datetime
import math
import os

import tensorflow as tf
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform import TFTransformOutput

import multiprocessing
N_CORES = multiprocessing.cpu_count()

tf.logging.set_verbosity(tf.logging.INFO)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def input_fn(tfrecords_path,
             tft_metadata,
             window_size,
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


def get_feature_spec(window_size):

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

    raw_feature_spec = get_feature_spec(window_size)

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

        # input_fn_train = input_fn(tfrecords_train_list,
        #                               tft_metadata,
        #                               args.window_size,
        #                               args.batch_size)

        # input_fn_eval = input_fn(tfrecords_eval_list,
        #                               tft_metadata,
        #                               args.window_size,
        #                               args.batch_size)

        # model.fit(x=input_fn_train, epochs=10, steps_per_epoch=steps_per_epoch_train)

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
            save_summary_steps=int(steps_per_epoch_train/5),
            save_checkpoints_steps=int(steps_per_epoch_train/5),
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

        model_estimator.export_savedmodel(
            export_dir_base=export_dir,
            serving_input_receiver_fn=serving_input_receiver_fn(tft_metadata,
                                                                args.window_size)
        )
