import argparse
import kfp

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=str, default='0.1', help='Pipeline package path')
    parser.add_argument('--host', type=str, default=None, help='K8s host')
    args = parser.parse_args()

    client = kfp.Client(host=args.host)
    result = client.upload_pipeline(pipeline_package_path=args.package)
    logger.info(result)
