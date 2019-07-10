import tensorflow as tf
import argparse


def get_rnn_model_sin_cos(window_size):
  
  # Define inputs
  hour_sin = tf.keras.Input(shape=[window_size,1], name='hour_sin')
  day_of_week_sin = tf.keras.Input(shape=[window_size,1], name='day_of_week_sin')
  day_of_month_sin = tf.keras.Input(shape=[window_size,1], name='day_of_month_sin')
  week_number_sin = tf.keras.Input(shape=[window_size,1], name='week_number_sin')
  month_sin = tf.keras.Input(shape=[window_size,1], name='month_sin')
  
  hour_cos = tf.keras.Input(shape=[window_size,1], name='hour_cos')
  day_of_week_cos = tf.keras.Input(shape=[window_size,1], name='day_of_week_cos')
  day_of_month_cos = tf.keras.Input(shape=[window_size,1], name='day_of_month_cos')
  week_number_cos = tf.keras.Input(shape=[window_size,1], name='week_number_cos')
  month_cos = tf.keras.Input(shape=[window_size,1], name='month_cos')
  
  n_trips = tf.keras.Input(shape=[window_size,1], name='n_trips')
  
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
    n_trips
 ]
   
  concat = tf.keras.layers.concatenate(input_tensors,axis=-1)
  
  net = tf.keras.layers.CuDNNGRU(16,return_sequences=False)(concat)
                            bias_initializer='glorot_uniform',
                            return_sequences=False)(concat)

  net = tf.keras.layers.Dropout(0.1)(net)

  net = tf.keras.layers.Dense(8, activation=tf.nn.relu,
                             bias_initializer='glorot_uniform',
                             )(net)
  net = tf.keras.layers.Dense(1,name='target')(net)
  
  return tf.keras.Model(inputs=input_tensors, outputs=net)


if __name__ == '__main__':
    ...