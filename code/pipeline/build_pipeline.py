# -*- coding: utf-8 -*-

import kfp.dsl as dsl
import kfp.gcp as gcp
import os


@dsl.pipeline(
    name='Time-Series-Forecast for Chicago Taxi dataset',
    description='A pipeline to preprocess, train and measure metrics for a time series forecast problem.'
)
def chicago_taxi_pipeline(
        artifacts_dir="gs://ciandt-cognitive-sandbox-chicago-taxi-demo-bucket/{{workflow.uid}}/artifacts",
        model_dir="gs://ciandt-cognitive-sandbox-chicago-taxi-demo-bucket/{{workflow.uid}}/models",
        project_id="ciandt-cognitive-sandbox",
        start_date="2019-04-01",
        end_date="2019-04-30",
        split_date="2019-04-20",
        window_size=24,  # hours
        model_name="rnn_v1",
        deployed_model_name="chicago_taxi_forecast",
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

    temp_dir = os.path.join(str(artifacts_dir), "temp")

    bq2tfrecord = dsl.ContainerOp(name='bq2tfrecord',
                                  image='gcr.io/ciandt-cognitive-sandbox/chicago-taxi-forecast/preproc:latest',
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
                                      "tft_artifacts_dir": "/tft_artifacts_dir.txt"
                                  }
                                  ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    train = dsl.ContainerOp(
        name='train',
        image='gcr.io/ciandt-cognitive-sandbox/chicago-taxi-forecast/train:latest',
        command=["python3", "/app/train.py"],
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
        },
        output_artifact_paths={
            'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'
        }
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(bq2tfrecord)  # .set_gpu_limit(1)

    deploy = dsl.ContainerOp(
        name='deploy',
        image='gcr.io/ciandt-cognitive-sandbox/chicago-taxi-forecast/deploy:latest',
        command=["python", "/app/deploy_cmle.py"],
        arguments=[
            "--project", project_id,
            "--gcs-path", train.outputs["saved_model_path"],
            "--model-name", deployed_model_name
        ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(train)  # .set_gpu_limit(1)

    predictions = dsl.ContainerOp(
        name='predictions',
        image='gcr.io/ciandt-cognitive-sandbox/chicago-taxi-forecast/evaluate:latest',
        command=["python", "/app/make_predictions.py"],
        arguments=[
            "--model-name", deployed_model_name,
            "--project", project_id,
            "--window-size", window_size,
            "--start-date", split_date,
            "--end-date", end_date,
            "--znorm-stats-json", bq2tfrecord.outputs['znorm_stats'],
            "--batch-size", prediction_batch_size,
            "--output-path", "gs://ciandt-cognitive-sandbox-chicago-taxi-demo-bucket/{{workflow.uid}}/predictions/forecast.csv"
        ],
        file_outputs={
            "prediction_csv_path": "/prediction_csv_path.txt"
        }
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(deploy)

    metrics = dsl.ContainerOp(
        name='metrics',
        image='gcr.io/ciandt-cognitive-sandbox/chicago-taxi-forecast/evaluate:latest',
        command=["python", "/app/evaluate.py"],
        arguments=[
            "--prediction-csv", predictions.outputs['prediction_csv_path']
        ],
        output_artifact_paths={
            "mlpipeline-metrics": "/mlpipeline-metrics.json"
        }
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(predictions)

    plot = dsl.ContainerOp(
        name='plot_time_series',
        image='gcr.io/ciandt-cognitive-sandbox/chicago-taxi-forecast/evaluate:latest',
        command=["python", "/app/plot_series.py"],
        arguments=[
            "--prediction-csv", predictions.outputs['prediction_csv_path'],
            "--output-dir", "gs://ciandt-cognitive-sandbox-chicago-taxi-demo-bucket/{{workflow.uid}}/plots"
        ]
    ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(predictions)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(chicago_taxi_pipeline,  'chicago_taxi_pipeline.tar.gz')
