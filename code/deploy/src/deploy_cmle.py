from tensorflow.python.lib.io import file_io
import argparse
import os
import subprocess
import datetime
import time
import yaml
import sys

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main(argv=None):
    parser = argparse.ArgumentParser(description='ML Deployment')
    parser.add_argument(
        '--project',
        help='The GCS project to use',
        required=True)
    parser.add_argument(
        '--gcs-path',
        help='The GCS path to the trained model. The path should end with "../export/<model-name>".',
        required=True)
    parser.add_argument(
        '--model-name',
        help='The model name.',
        default='chicago_taxi_forecast')
    parser.add_argument(
        '--region',
        help='The model region.',
        default='us-central1'
    )

    args = parser.parse_args()

    # # Make sure the model dir exists before proceeding, as sometimes it takes a few seconds to become
    # # available after training completes.
    # retries = 0
    # sleeptime = 5
    # while retries < 20:
    #     try:
    #         model_location = os.path.join(
    #             args.gcs_path, file_io.list_directory(args.gcs_path)[-1])
    #         print("model location: %s" % model_location)
    #         break
    #     except Exception as e:
    #         print(e)
    #         print("Sleeping %s seconds to wait for GCS files..." % sleeptime)
    #         time.sleep(sleeptime)
    #         retries += 1
    #         sleeptime *= 2
    # if retries >= 20:
    #     print("could not get model location subdir from %s, exiting" %
    #           args.gcs_path)
    #     exit(1)

    # Check if the model already exists
    model_describe_command = ['gcloud', 'ai-platform', 'models',
                              'describe', args.model_name, '--project', args.project]

    region = None
    new_version = datetime.datetime.now().strftime('v%Y%m%d_%H%M%S')
    try:
        result = subprocess.check_output(model_describe_command)
        logger.info(result)
        model_info = yaml.load(result)
        region = model_info['regions']
        old_version = model_info['defaultVersion']['name'].split('/')[-1]

        logger.info(
            """
            Model '{}' already exist at region(s) {}. 
            Updating version from {} to {}
            """.format(args.model_name, region, old_version, new_version))

    except subprocess.CalledProcessError as e:
        region = args.region
        logger.info("""
        Model '{}' doesn't exist.
        Deploy at region {} version {}
        """.format(args.model_name, region, new_version))

        model_create_command = ['gcloud', 'ai-platform', 'models',
                                'create', args.model_name, '--regions',
                                region, '--project', args.project]
        logger.info(model_create_command)
        result = subprocess.call(model_create_command)
        logger.info(result)

    model_deploy_command = ['gcloud', 'ai-platform', 'versions', 'create', new_version,
                            '--model', args.model_name, '--runtime-version', '1.13',
                            '--python-version', '3.5',
                            '--project', args.project, '--origin', args.gcs_path
                            ]
    logger.info(model_deploy_command)
    result = subprocess.call(model_deploy_command)
    logger.info(result)

    logger.info('Set {} as default'.format(new_version))

    model_deploy_command = ['gcloud', 'ai-platform', 'versions', 'set-default', new_version,
                            '--model', args.model_name, '--project', args.project
                            ]
    result = subprocess.call(model_deploy_command)
    logger.info(result)


if __name__ == "__main__":
    main()
