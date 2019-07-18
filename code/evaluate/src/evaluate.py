# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import json

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("--prediction-csv",
                        dest="prediction_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.prediction_csv)
    mse_norm = mean_squared_error(df['target_norm'], df['prediction_norm'])
    mse = mean_squared_error(df['target'], df['prediction'])

    metrics = {
        'metrics': [{
            'name': 'mse-norm',
            'numberValue':  mse_norm,
            'format': "RAW",
        }, {
            'name': 'mse',
            'numberValue':  mse,
            'format': "RAW",
        }]
    }    

    with open('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
