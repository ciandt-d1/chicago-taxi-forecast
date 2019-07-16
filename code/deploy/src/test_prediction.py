# -*- coding: utf-8 -*-

import tensorflow as tf
import base64
import googleapiclient.discovery
import sys
import tqdm
import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict_json(project, model, service, instances):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.        
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    name = 'projects/{}/models/{}'.format(project, model)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':

    example = tf.train.Example(features=tf.train.Features(feature={
        'hour': int_feature([2, 2, 2, 2, 3, 3]),
        'day_of_week': int_feature([2, 2, 2, 2, 2, 2]),
        'day_of_month': int_feature([12, 12, 12, 12, 12, 12]),
        'week_number': int_feature([3, 3, 3, 3, 3, 3]),
        'month': int_feature([1, 1, 1, 1, 1, 1]),
        'community_area': int_feature([8, 8, 8, 8, 8, 8]),
        'n_trips': float_feature([100, 33, 44, 66, 77, 44]),
        'community_area_code': int_feature([8])
    })).SerializeToString()

    example = base64.urlsafe_b64encode(example)

    model_name = "chicago_taxi_forecast"

    service = googleapiclient.discovery.build(
        'ml', 'v1', cache_discovery=False)

    try:
        predictions = predict_json(
            "ciandt-cognitive-sandbox", "chicago_taxi_forecast", service, example)
        logger.info("Prediction works: {}".format(predictions))
    except RuntimeError as e:
        logger.error("An error occurred: \n{}".format(e))
        sys.exit(1)

    logger.info("Assessing optimal batch_size")    
    batch_size_list = [128, 256, 512, 1024,2048]

    n_predictions = 4096*14

    try:
        for batch_size in batch_size_list:
            n_batches = int(n_predictions // batch_size)
            avg_time = 0
            for i in tqdm.tqdm(range(n_batches)):
                start_time = time.time()
                predictions = predict_json(
                    "ciandt-cognitive-sandbox", "chicago_taxi_forecast", service, [example]*batch_size)
                end_time = time.time()
                avg_time += end_time - start_time
            avg_time /= n_batches
            avg_time /= batch_size
            logger.info("Batch size {} takes {}s to predict each tf.example".format(
                batch_size, avg_time))
    except RuntimeError as e:
        logger.error("An error occurred: \n{}".format(e))
        sys.exit(1)
