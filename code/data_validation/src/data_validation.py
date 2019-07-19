import argparse
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft

from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import metadata_io

import json

from google.protobuf import text_format
import google.cloud.storage

from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2

import os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data-path',
                        dest='input_data_path', type=str, required=True)
    parser.add_argument('--output-dir',
                        dest='output_dir', type=str, required=True)
    args = parser.parse_args()

    stats = tfdv.generate_statistics_from_tfrecord(
        data_location=args.input_data_path)
    inferred_schema = tfdv.infer_schema(statistics=stats)

    logger.info("Inferred schema \n {}".format(inferred_schema))
    schema_path = os.path.join(args.output_dir, 'schema.pb')

    schema_text = text_format.MessageToString(inferred_schema)
    file_io.write_string_to_file(schema_path, schema_text)

    tfdv_html = tfdv.utils.display_util.get_statistics_html(stats)

    static_html_path = os.path.join(args.output_dir, 'web_app.html')
    logger.info("static_html_path {}".format(static_html_path))

    # remove HTML iframe tags, otherwise the html won't render on kubeflow UI
    start_str="<link rel"
    end_str = "</facets-overview>"
    start_str_i = tfdv_html.index(start_str)
    end_str_i = tfdv_html.index(end_str)+len(end_str)
    tfdv_html=tfdv_html[start_str_i:end_str_i]
    
    file_io.write_string_to_file(static_html_path, tfdv_html)

    metadata = {
        'outputs': [{
            'type': 'web-app',
            'storage': 'gcs',
            'source': static_html_path,
        }]
    }
    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    with file_io.FileIO('/schema.txt', 'w') as f:
        f.write(schema_path)
