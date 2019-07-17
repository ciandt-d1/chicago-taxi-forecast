# -*- coding: utf-8 -*-

from tensorflow.python.lib.io import file_io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import argparse
import matplotlib as mpl
mpl.use('Agg')

import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("--prediction-csv",
                        dest="prediction_csv", type=str, required=True)
    parser.add_argument("--output-dir",
                        dest="output_dir", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.prediction_csv)

    for community_area, ca_df in tqdm.tqdm(df.groupby("community_area")):
        fig_file = file_io.FileIO(os.path.join(
            args.output_dir, 'ca_{}.png'.format(community_area)), "w")

        sorted_df = ca_df.sort_values(['date', 'hour'], ascending=True)
        plt.figure(figsize=(20, 5))
        plt.title("Community Area {}".format(community_area))
        plt.plot(sorted_df["target"].values, label='target')
        plt.plot(sorted_df["prediction"].values, label='prediction')
        plt.legend()
        plt.savefig(fig_file)
        plt.close()
