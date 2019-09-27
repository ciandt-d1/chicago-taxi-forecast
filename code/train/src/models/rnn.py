# -*- coding: utf-8 -*-

""" RNN models
"""

import tensorflow as tf
import math

def rnn_v1(window_size, n_areas=77):
    """
    Gated Recurrent Unit (GRU) NN.

    OBS: CMLE does not support CuDNN layers, 
    so tf.keras.layers.GRU is being used, instead of tf.keras.layers.CuDNNGRU
    """
    
    hour_sin = tf.keras.Input(shape=[window_size, 1], name='hour_sin')
    day_of_week_sin = tf.keras.Input(
        shape=[window_size, 1], name='day_of_week_sin')
    day_of_month_sin = tf.keras.Input(
        shape=[window_size, 1], name='day_of_month_sin')
    week_number_sin = tf.keras.Input(
        shape=[window_size, 1], name='week_number_sin')
    month_sin = tf.keras.Input(shape=[window_size, 1], name='month_sin')

    hour_cos = tf.keras.Input(shape=[window_size, 1], name='hour_cos')
    day_of_week_cos = tf.keras.Input(
        shape=[window_size, 1], name='day_of_week_cos')
    day_of_month_cos = tf.keras.Input(
        shape=[window_size, 1], name='day_of_month_cos')
    week_number_cos = tf.keras.Input(
        shape=[window_size, 1], name='week_number_cos')
    month_cos = tf.keras.Input(shape=[window_size, 1], name='month_cos')

    community_area = tf.keras.Input(shape=[window_size], name='community_area')

    community_area_embedding_size = math.ceil(math.pow(n_areas, 0.25)) + 1

    community_area_embedding = tf.keras.layers.Embedding(
        input_dim=n_areas+1, output_dim=community_area_embedding_size, input_length=window_size)(community_area)

    n_trips = tf.keras.Input(shape=[window_size, 1], name='n_trips')

    input_tensors = [
        hour_sin,
        day_of_week_sin,
        day_of_month_sin,
        week_number_sin,
        month_sin,
        hour_cos,
        day_of_week_cos,
        day_of_month_cos,
        week_number_cos,
        month_cos,
        community_area,
        n_trips
    ]

    tensors_to_concat = [
        hour_sin,
        day_of_week_sin,
        day_of_month_sin,
        week_number_sin,
        month_sin,
        hour_cos,
        day_of_week_cos,
        day_of_month_cos,
        week_number_cos,
        month_cos,
        n_trips,
        community_area_embedding
    ]

    concat = tf.keras.layers.concatenate(tensors_to_concat, axis=-1)

    # net = tf.keras.layers.CuDNNGRU(16, return_sequences=False)(concat)
    net = tf.keras.layers.GRU(16, return_sequences=False)(concat)

    net = tf.keras.layers.Dropout(0.1)(net)

    net = tf.keras.layers.Dense(8, activation=tf.nn.relu,
                                bias_initializer='glorot_uniform',
                                )(net)
    net = tf.keras.layers.Dense(1, name='target')(net)

    return tf.keras.Model(inputs=input_tensors, outputs=net)
