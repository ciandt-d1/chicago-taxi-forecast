# Preprocessing module

## Build for Kubeflow pipeline
In any doubt, check the [official documentation](https://www.kubeflow.org/docs/gke/gcp-e2e/)


### Build Image
```
export DEPLOYMENT_NAME=chicago-taxi-forecast
export PROJECT=ciandt-cognitive-sandbox
export GCP_SERVICE_ACC_JSON_PATH=ts-forecast-vm@ciandt-cognitive-sandbox.iam.gserviceaccount.com # to access google cloud storage
export VERSION_TAG=latest
export DOCKER_IMAGE_NAME=gcr.io/${PROJECT}/${DEPLOYMENT_NAME}-preproc:${VERSION_TAG}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile
```

### Test locally

```
docker run -it -v ${PWD}/src:/src --rm  ${DOCKER_IMAGE_NAME} bash
```

Run container
```
python3 bq2tfrecord.py \
--tfrecord-dir /tmp \
--tfx-artifacts-dir /tmp \
--project ciandt-cognitive-sandbox \
--window-size 6 \
--start-date 2019-04-10 \
--end-date  2019-04-30 \
--split-date 2019-04-20 \
--temp-dir /tmp \
--runner DirectRunner
```

### Run directly on Google Cloud DataFlow

Build package

```

```

Submit job
```

```

### Upload container image to the GCP Conainer Registry
```
gcloud auth configure-docker --quiet

docker push ${DOCKER_IMAGE_NAME}
```
