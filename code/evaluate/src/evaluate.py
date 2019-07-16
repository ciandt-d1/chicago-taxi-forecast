# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import json
import googleapiclient.discovery
import base64
import sys
import tensorflow as tf

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def query_trips(start_date, end_date):

    query_str = """
        SELECT
            pickup_community_area,
            EXTRACT(DATE from trip_start_timestamp) as date,
            EXTRACT(HOUR from trip_start_timestamp) as hour,
            COUNT(*) as n_trips
                FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                WHERE trip_start_timestamp >= '{start_date}'
                AND trip_start_timestamp <'{end_date}'
                AND pickup_community_area is NOT NULL
                AND trip_start_timestamp is NOT NULL
        GROUP BY date, hour, pickup_community_area
        ORDER BY date, hour, pickup_community_area ASC
    """.format(start_date=start_date, end_date=end_date)

    return query_str


def make_prediction(model_url, service, instances):

    response = service.projects().predict(
        name=model_url,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == "__main__":  

    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("--model-name", dest="model_name",
                        type=str, required=True)
    parser.add_argument("--project", dest="project", type=str, required=True)
    parser.add_argument("--window-size", dest="window_size",
                        type=int, required=True)
    parser.add_argument("--start-date", dest="start_date",
                        type=str, required=True)
    parser.add_argument("--end-date", dest="end_date", type=str, required=True)
    parser.add_argument("--znorm-stats-json",
                        dest="znorm_stats_json", type=str, required=True)
    parser.add_argument("--batch-size",
                        dest="batch_size", type=int, required=True)
    args = parser.parse_args()

    model_url = 'projects/{}/models/{}'.format(args.project, args.model_name)

    # build CMLE connector
    service = googleapiclient.discovery.build(
        'ml', 'v1', cache_discovery=False)

    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'hour': int_feature([2, 2, 2, 2, 3, 3]),
    #     'day_of_week': int_feature([2, 2, 2, 2, 2, 2]),
    #     'day_of_month': int_feature([12, 12, 12, 12, 12, 12]),
    #     'week_number': int_feature([3, 3, 3, 3, 3, 3]),
    #     'month': int_feature([1, 1, 1, 1, 1, 1]),
    #     'community_area': int_feature([8, 8, 8, 8, 8, 8]),
    #     'n_trips': float_feature([100, 33, 44, 66, 77, 44]),
    #     'community_area_code': int_feature([8])
    # })).SerializeToString()    

    # example = base64.urlsafe_b64encode(example).decode('utf-8')

    # logger.info(example)
    # logger.info(type(example))
    
    # # logger.info("type(example) {}".format(type(example)))
    
    # # logger.info(example)
    # predictions = make_prediction(model_url, service, example)
    # logger.info(predictions)
    # sys.exit(0)
    

    # Load community areas mean and std to reverse znorm
    znorm_stats = json.load(open(args.znorm_stats_json, 'r'))
    znorm_stats = {int(ca): {'mean': mean, 'std': std} for ca, mean, std in zip(
        znorm_stats['pickup_community_area'], znorm_stats['mean'], znorm_stats['std'])}

    # Query BQ
    query = query_trips(args.start_date, args.end_date)
    df = pd.read_gbq(query, dialect='standard')

    # Extract temporal features
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_numeric(df['hour'])
    df['day_of_month'] = df['date'].apply(
        lambda t: t.day)
    df['day_of_week'] = df['date'].apply(
        lambda t: t.dayofweek)
    df['month'] = df['date'].apply(lambda t: t.month)
    df['week_number'] = df['date'].apply(
        lambda t: t.weekofyear)
    logger.info(df.head())

    predictions_dict = {
        "community_area": [],
        "target": [],
        "prediction_norm": [],
        "date": []
    }

    batch_buffer = []
    mean_buffer = []
    std_buffer = []

    batch_i = 0
    n_batches = int((len(df) // args.batch_size))

    for ca, trips_time_series in df.groupby('pickup_community_area'):

        # force sorting
        ts_df = trips_time_series.sort_values(['date', 'hour'], ascending=True)        
        n_windows = len(ts_df)-args.window_size-1
        

        for i in range(0, n_windows, 1):
            window = ts_df.iloc[i:(i+args.window_size+1)]

            ca_code = window['pickup_community_area'].tolist()[0]            

            example = tf.train.Example(features=tf.train.Features(feature={
                'hour': int_feature(window['hour'][:args.window_size].tolist()),
                'day_of_week': int_feature(window['day_of_week'][:args.window_size].tolist()),
                'day_of_month': int_feature(window['day_of_month'][:args.window_size].tolist()),
                'week_number': int_feature(window['week_number'][:args.window_size].tolist()),
                'month': int_feature(window['month'][:args.window_size].tolist()),
                'community_area': int_feature(window['pickup_community_area'][:args.window_size].tolist()),
                'n_trips': float_feature(window['n_trips'][:args.window_size].tolist()),
                'community_area_code': int_feature([ca_code])
            })).SerializeToString()

            example_b64 = base64.urlsafe_b64encode(example).decode('utf-8')

            batch_buffer.append(example_b64)
            mean_buffer.append(znorm_stats[ca]['mean'])
            std_buffer.append(znorm_stats[ca]['std'])

            predictions_dict['community_area'].append(ca)
            predictions_dict['date'].append(window['date'].values[args.window_size])
            predictions_dict['target'].append(window['n_trips'].values[args.window_size])
            

            if len(batch_buffer) == args.batch_size:
                predictions = make_prediction(model_url, service, batch_buffer)
                predictions = [v['target'][0] for v in predictions]
                predictions_dict['prediction_norm'].extend(predictions)
                batch_buffer.clear()
                
                logger.info("Batch {} of {} !".format(batch_i, n_batches))
                batch_i+=1
            
