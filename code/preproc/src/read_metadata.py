# -*- coding: utf-8 -*-

"""
Retrieve metadata from BigQuery
"""

import argparse
import pandas as pd
import numpy as np
import datetime
import os
import sys
import tensorflow as tf
import json
import apache_beam as beam
import tensorflow_transform as tft
from tensorflow.python.lib.io import file_io

from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io, beam_metadata_io


try:
    try:
        from apache_beam.options.pipeline_options import PipelineOptions
    except ImportError:
        from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
    from apache_beam.utils.options import PipelineOptions

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

COMMUNITY_AREA_QUERY = """
    SELECT DISTINCT pickup_community_area FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE pickup_community_area IS NOT NULL
"""


def _query_znorm_stats(start_date, end_date):
    """ Private function which returns a query string """
    query_str = """
        SELECT pickup_community_area, AVG(n_trips) mean, STDDEV(n_trips)+1 std FROM
            (SELECT
                pickup_community_area,
                EXTRACT(DATE from trip_start_timestamp) as date,
                EXTRACT(HOUR from trip_start_timestamp) as hour,
                COUNT(*) as n_trips
                FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                    WHERE trip_start_timestamp >= '{start_date}'
                    AND trip_start_timestamp < '{end_date}'
                    AND pickup_community_area IS NOT NULL
                GROUP BY date, hour, pickup_community_area
                ORDER BY date, hour ASC)
        GROUP BY pickup_community_area
        ORDER BY pickup_community_area ASC
    """.format(start_date=start_date, end_date=end_date)

    return query_str


class ParseZnormStatsRow(beam.DoFn):
    """ Parse BigQuery Row """

    def process(self, element):
        yield {'pickup_community_area': str(element['pickup_community_area']),
               'mean': float(element['mean']),
               'std': float(element['std'])
               }


def _read_znorm_stats_from_bq(pipeline, start_date, end_date):
    """ Read z-norm stats """

    query_str = _query_znorm_stats(start_date, end_date)
    raw_data = (pipeline |
                "Read znorm stats from BigQuery from {} to {}".format(start_date, end_date) >> beam.io.Read(
                    beam.io.BigQuerySource(query=query_str, use_standard_sql=True)) |
                "Parse stats from {} to {}".format(start_date, end_date) >> beam.ParDo(ParseZnormStatsRow()))

    return raw_data


class CombineZnormStats(beam.CombineFn):
    """ Group z-norm stats for each chicago community area """

    def __init__(self):
        super(CombineZnormStats, self).__init__()

    def create_accumulator(self):
        return {}

    def add_input(self, accumulator, element):
        # accumulator.append(str(element.get('pickup_community_area')))
        for k in element.keys():
            if k not in accumulator:
                accumulator[k] = []
            accumulator[k].append(element[k])

        return accumulator

    def merge_accumulators(self, accumulators):
        output = {}
        for acc in accumulators:
            for k in acc:
                if k not in output:
                    output[k] = []
                output[k].extend(acc[k])

        return output

    def extract_output(self, output):
        return output


class CombineCommunityArea(beam.CombineFn):
    """ Group each chicago community area """

    def __init__(self):
        super(CombineCommunityArea, self).__init__()

    def create_accumulator(self):
        return []

    def add_input(self, accumulator, element):
        accumulator.append(str(element.get('pickup_community_area')))
        return accumulator

    def merge_accumulators(self, accumulators):
        output = []
        for a in accumulators:
            output.extend(a)
        return output

    def extract_output(self, output):
        return list(set(output))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()    
    parser.add_argument('--tfx-artifacts-dir',
                        dest='tft_artifacts_dir', required=True)
    parser.add_argument('--project', dest='project', required=True)
    parser.add_argument('--start-date', dest='start_date', required=True)
    parser.add_argument('--end-date', dest='end_date', required=True)
    parser.add_argument('--split-date', dest='split_date', required=True)
    parser.add_argument('--temp-dir', dest='temp_dir',
                        required=False, default='/tmp')
    known_args, pipeline_args = parser.parse_known_args()

    pipeline_args.append('--project')
    pipeline_args.append(known_args.project)

    start_datetime = datetime.datetime.strptime(
        known_args.start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(known_args.end_date, '%Y-%m-%d')
    split_datetime = datetime.datetime.strptime(
        known_args.split_date, '%Y-%m-%d')

    datetime_now_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    if 'DataflowRunner' in pipeline_args:
        pipeline_args.extend(['--project', known_args.project,
                              '--temp_location', known_args.temp_dir,
                              '--save_main_session',
                              '--setup_file', './setup.py',
                              '--job_name', 'chicago-taxi-trips-read-metadata-{}'.format(datetime_now_str)])

    logger.info(pipeline_args)

    pipeline_options = PipelineOptions(flags=pipeline_args)

    community_area_list_path = os.path.join(
        known_args.tft_artifacts_dir, 'community_area_list.json')
    znorm_stats_path = os.path.join(
        known_args.tft_artifacts_dir, 'znorm_stats.json')

    # Query metadata
    with beam.Pipeline(options=pipeline_options) as pipeline:

        # List eligible community areas
        community_area_list_p = (pipeline |
                                 'Query eligible Community Areas' >> beam.io.Read(beam.io.BigQuerySource(
                                     query=COMMUNITY_AREA_QUERY, use_standard_sql=True)) |
                                 'Combine list' >> beam.CombineGlobally(CombineCommunityArea()) |
                                 'Map to string' >> beam.Map(lambda l: ','.join(l)) |
                                 'Dump to text' >> beam.io.WriteToText(
                                     community_area_list_path, shard_name_template='')
                                 )

        # Query znorm statistics
        znorm_stats_p = _read_znorm_stats_from_bq(
            pipeline, known_args.start_date, known_args.split_date)

        _ = (znorm_stats_p |
             "Combine znorm Stats" >> beam.CombineGlobally(CombineZnormStats()) |
             "Map znorm stats to json" >> beam.Map(lambda d: json.dumps(d)) |
             "Dump znorm stats to file" >> beam.io.WriteToText(
                 znorm_stats_path, shard_name_template='')
             )

    with open("/community_area_list_path.txt", "w") as f:
        f.write(community_area_list_path)

    with open("/znorm_stats_path.txt", "w") as f:
        f.write(znorm_stats_path)
