# Evaluation module

This module evalutes a deployed model and builds timeseries plots as html widget to be
loaded at Kubeflow UI

## Build for Kubeflow pipeline
In any doubt, check the [official documentation](https://www.kubeflow.org/docs/gke/gcp-e2e/)


### Build Image
```
export DEPLOYMENT_NAME=chicago-taxi-forecast
export PROJECT=ciandt-cognitive-sandbox
export VERSION_TAG=latest
export DOCKER_IMAGE_NAME=gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/evaluate:${VERSION_TAG}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile
```

### Test locally

Define a local directory to read/write pipeline artifacts

```
ARTIFACTS_DIR=/home/CIT/rodrigofp/Projects/Specialization2019/demo_taxi/assets
docker run -it -v ${PWD}/src:/src -v ${ARTIFACTS_DIR}:/artifacts_dir --rm  ${DOCKER_IMAGE_NAME} bash
```

Run container:

First, make predictions
```
python3 make_predictions.py \
--model-name chicago_taxi_forecast \
--project ciandt-cognitive-sandbox \
--window-size 6 \
--start-date 2019-04-20 \
--end-date  2019-04-30 \
--znorm-stats-json /artifacts_dir/znorm_stats.json \
--batch-size 512 \
--output-path /artifacts_dir/predictions.csv
```

Measure evaluation metrics
```
python3 evaluate.py \
--prediction-csv /artifacts_dir/predictions.csv
```

Plot series
```
python3 plot_series.py \
--prediction-csv /artifacts_dir/predictions.csv \
--output-dir /artifacts_dir/plots
```

### Upload container image to the GCP Conainer Registry
```
gcloud auth configure-docker --quiet

docker push ${DOCKER_IMAGE_NAME}
```
