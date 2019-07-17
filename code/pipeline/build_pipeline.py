# -*- coding: utf-8 -*-

import kfp.dsl as dsl
import kfp.gcp as gcp
import os


@dsl.pipeline(
    name='Time-Series-Forecast for Chicago Taxi dataset',
    description='A pipeline to preprocess, train and measure metrics for a time series forecast problem.'
)
def chicago_taxi_pipeline(
        artifacts_dir="gs://ciandt-cognitive-sandbox-ts-forecast-bucket/data/timeseries",
        model_dir="gs://ciandt-cognitive-sandbox-ts-forecast-bucket/models",
        project_id="ciandt-cognitive-sandbox",
        start_date="2019-04-01",
        end_date="2019-04-30",
        split_date="2019-04-20",
        window_size=24,  # hours
        model_name="rnn_v1",
        dataflow_runner="DirectRunner",
        epochs=10,
        train_batch_size=128,
        prediction_batch_size=512,
        gpu_mem_usage="0.9"):
    """
    Pipeline with 7 stages:
      1. Extract and transform input data from BigQuery into tfrecords
      2. Data Validation
      3. Train NN
      4. Deploy NN on CMLE
      5. Make predictions
      6. Evaluation
      7. Plot time series
    """

    temp_dir = os.path.join(artifacts_dir, "temp")

    bq2tfrecord = dsl.ContainerOp(name='bq2tfrecord',
                                  image='gcr.io/{project}/chicago-taxi-forecast-cluster/preproc:latest'.format(
                                      project=project_id),
                                  command=["python3", "/app/bq2tfrecord.py"],
                                  arguments=[
                                      "--tfrecord-dir", artifacts_dir,
                                      "--tfx-artifacts-dir", artifacts_dir,
                                      "--project", project_id,
                                      "--window-size", window_size,
                                      "--start-date", start_date,
                                      "--end-date", end_date,
                                      "--split-date", split_date,
                                      "--temp-dir", temp_dir,
                                      "--runner", dataflow_runner
                                  ],
                                  file_outputs={
                                      "train_tfrecord_path": "/train_tfrecord_path.txt",
                                      "eval_tfrecord_path": "/eval_tfrecord_path.txt",
                                      "znorm_stats": "/znorm_stats.txt",
                                      "n_areas": "/n_areas.txt",
                                      "n_windows_train": "/n_windows_train.txt",
                                      "n_windows_eval": "/n_windows_eval.txt",
                                      "tft_artifacts_dir": "tft_artifacts_dir.txt"
                                  }
                                  ).apply(gcp.use_gcp_secret('user-gcp-sa'))


    train = dsl.ContainerOp(
        name='train',
        image='gcr.io/{project}/chicago-taxi-forecast-cluster/train:latest'.format(
            project=project_id),
        command=["python", "/app/train_rnn_multi_sku.py"],
        arguments=[
            "--tfrecord-file-train", bq2tfrecord.outputs["train_tfrecord_path"],
            "--tfrecord-file-eval", bq2tfrecord.outputs["eval_tfrecord_path"],
            "--tft-artifacts-dir", bq2tfrecord.outputs["tft_artifacts_dir"],
            "--model-name", model_name,
            "--n-windows-train", bq2tfrecord.outputs["n_windows_train"],
            "--n-windows-eval", bq2tfrecord.outputs["n_windows_eval"],
            "--window-size", window_size,
            "--n-areas", bq2tfrecord.outputs["n_areas"],
            "--epochs", epochs,
            "--batch-size", train_batch_size,
            "--output-dir", model_dir,
            "--gpu-memory-fraction", gpu_mem_usage
        ],
        file_outputs={
            "saved_model_path": "/saved_model_path.txt"
        }
    ).set_gpu_limit(1).apply(gcp.use_gcp_secret('user-gcp-sa')).after(bq2tfrecord)

# deploy = dsl.ContainerOp(
#     name='predict',
#     image='gcr.io/ciandt-cognitive-sandbox/ts-forecast-train:latest',
#     command=["python", "/app/predict_multi_sku.py"],
#     arguments=[
#         "--model-name", model_name,
#         "--ckpt-path", train.outputs["ckpt_path"],
#         "--metadata-tfrecord-file", csv2tfrecord.outputs["metadata_filename"],
#         "--split-date", split_date,
#         "--timeseries-path", bq2csv.outputs['output_dir'],
#         "--gpu-memory-fraction", gpu_mem_usage
#     ],
#     file_outputs={
#         "prediction_path": "/prediction_path.txt"
#     }
# ).set_gpu_limit(1).apply(gcp.use_gcp_secret('user-gcp-sa')).after(train)

# metrics = dsl.ContainerOp(
#     name='metrics',
#     image='gcr.io/ciandt-cognitive-sandbox/ts-forecast-metrics:latest',
#     command=["python", "/app/metrics.py"],
#     arguments=[
#         "--prediction-file", predict.outputs['prediction_path'],
#         "--metadata-tfrecord-file", csv2tfrecord.outputs["metadata_filename"]
#     ],
#     file_outputs={
#         "mlpipeline-metrics": "/mlpipeline-metrics.json"
#     }
# ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(predict)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(chicago_taxi_pipeline, __file__ + '.tar.gz')
