# -*- coding: utf-8 -*-

""" Tensorflow model zoo
    Please, register any new model at MODEL_ZOO dictionary
"""

from models import rnn

import sys
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_ZOO = {
    "rnn_v1": rnn.rnn_v1    
}


def get_model(model_name, **kw):
    """ Get a Tensorflow model """
    if model_name not in MODEL_ZOO:
        logger.error('Model {} does not exist')
        sys.exit(1)

    logger.info('Loading model {}'.format(model_name))
    return MODEL_ZOO[model_name](**kw)
