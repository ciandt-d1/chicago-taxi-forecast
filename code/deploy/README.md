# Deployment module

Currently the ML models is being deployed on Google Cloud ML Engine

## Build for Kubeflow pipeline
In any doubt, check the [official documentation](https://www.kubeflow.org/docs/gke/gcp-e2e/)


### Build Image
```
export DEPLOYMENT_NAME=chicago-taxi-forecast
export PROJECT=ciandt-cognitive-sandbox
export VERSION_TAG=latest
export DOCKER_IMAGE_NAME=gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/deploy:${VERSION_TAG}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile
```

### Test locally

```
ARTIFACTS_DIR=/home/CIT/rodrigofp/Projects/Specialization2019/demo_taxi/assets

docker run -it -v ${PWD}/src:/src --rm  ${DOCKER_IMAGE_NAME} bash
docker run -it -v ${PWD}/src:/src -v ${ARTIFACTS_DIR}:/artifacts_dir --rm  ${DOCKER_IMAGE_NAME} bash

python deploy_cmle.py \
--project ciandt-cognitive-sandbox \
--gcs-path gs://ciandt-cognitive-sandbox-ts-forecast-bucket/test/1563373565/ \
--model-name chicago_taxi_forecast

```


### Upload container image to the GCP Conainer Registry
```
gcloud auth configure-docker --quiet

docker push ${DOCKER_IMAGE_NAME}
```
