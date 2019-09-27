# End-to-End time series forecast pipeline with Kubeflow and TFX

This repo contains the minimum steps to create an e2e time series forecast pipeline on KubeFlow and TFX deployed at Google Cloud Platform

The [chicago taxi rides dataset](https://digital.cityofchicago.org/index.php/chicago-taxi-data-released/) was used throughout this tutorial.

Steps covered in this tutorial in the suggested order:

*  How to create and deploy a Kubeflow cluster at GCP ([`kubeflow_cluster`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/kubeflow_cluster))
* Transform data from BigQuery at scale with [Tensorflow Transform](https://www.tensorflow.org/tfx/transform/get_started) ([`code/preproc`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/code/data_validation))
* Check for data anomalies and skewness with [Tensorflow Data Validation](https://www.tensorflow.org/tfx/data_validation/get_started) ([`code/data_validation`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/code/data_validation))
* Train model at K8s cluster ([`code/train`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/code/train))
* Deploy model at [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/)  ([`code/deploy`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/code/deploy))
* Measure model performance ([`code/evaluate`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/code/evaluate))
* Build and run [kubeflow pipeline](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/) ([`code/pipeline`](https://github.com/ciandt-d1/chicago-taxi-forecast/tree/master/code/pipeline))

In any doubts, please, contact:

* Diego Silva - diegosilva@ciandt.com
* Gabriel Moreira - gabrielpm@ciandt.com
* Leandro Vendramin - vendramin@ciandt.com
* Pedro Lelis - pedrolelis@ciandt.com
* Rodrigo Pereira - rodrigofp@ciandt.com