# -*- coding: utf-8 -*-

import logging
from tensorflow.python.lib.io import file_io
from google.cloud import storage
import os
import sys

import argparse
import pandas as pd
import json
import tqdm
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("--prediction-csv",
                        dest="prediction_csv", type=str, required=True)
    parser.add_argument("--output-dir",
                        dest="output_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.prediction_csv)

    if args.output_dir.startswith('gs://'):
        storage_client = storage.Client()
        bucket_name = args.output_dir.split('gs://')[-1].split('/')[0]
        bucket = storage_client.get_bucket(bucket_name)

    widget_html = """
    <html>
    <body>    
    """

    image_list = []

    for community_area, ca_df in tqdm.tqdm(df.groupby("community_area")):
        image_path = os.path.join(
            args.output_dir, 'ca_{}.png'.format(community_area))
        fig_file = file_io.FileIO(image_path, "w")

        sorted_df = ca_df.sort_values(['date', 'hour'], ascending=True)
        plt.figure(figsize=(20, 5))
        plt.title("Community Area {}".format(community_area))
        plt.plot(sorted_df["target"].values, label='target')
        plt.plot(sorted_df["prediction"].values, label='prediction')
        plt.legend()
        plt.savefig(fig_file)
        plt.close()

        # make images public on GCS
        if args.output_dir.startswith('gs://'):
            blob_name = '/'.join(image_path.split('gs://')[-1].split('/')[1:])
            image_uri = "https://storage.googleapis.com/{}/{}".format(
                bucket_name, blob_name)
        else:
            image_uri = image_path

        image_list.append(image_uri)

    max_tries = 5
    sleep_time = 10
    for image_uri in image_list:

        if image_uri.startswith('https://'):
            retries = 0

            while retries < max_tries:
                try:
                    blob_name = image_uri.split('storage.googleapis.com/{}/'.format(bucket_name))[1]
                    blob = bucket.blob(blob_name)
                    blob.make_public()
                    break
                except Exception as e:

                    logger.info("Sleeping {} secs to wait for GCS to upload {}".format(
                        sleep_time, blob_name))
                    logger.info(e)
                    time.sleep(sleep_time)                    
                    retries += 1
                    if retries >= max_tries:
                        logger.error("Could not locate {}" .format(image_uri))
                        # sys.exit(1)
                        break

        # append image to widget
        widget_html += """
        <img src="{}">
        """.format(image_uri)

    widget_html += """
    </body>
    </html>    
    """

    static_html_path = os.path.join(args.output_dir, "web_app.html")
    file_io.write_string_to_file(static_html_path, widget_html)

    metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'gcs',
            'source': static_html_path,
        }]
    }
    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
