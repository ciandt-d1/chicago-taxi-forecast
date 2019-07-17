# Training module

## Build for Kubeflow pipeline
In any doubt, check the [official documentation](https://www.kubeflow.org/docs/gke/gcp-e2e/)


### Build Image
```
export DEPLOYMENT_NAME=chicago-taxi-forecast
export PROJECT=ciandt-cognitive-sandbox
export GCP_SERVICE_ACC_JSON_PATH=ts-forecast-vm@ciandt-cognitive-sandbox.iam.gserviceaccount.com # to access google cloud storage
export VERSION_TAG=latest
export DOCKER_IMAGE_NAME=gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/train:${VERSION_TAG}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile
```

### Test locally

Define a local directory to read/write pipeline artifacts

```
ARTIFACTS_DIR=/home/CIT/rodrigofp/Projects/Specialization2019/demo_taxi/assets
docker run -it -v ${PWD}/src:/src -v ${ARTIFACTS_DIR}:/artifacts_dir --rm --runtime=nvidia  ${DOCKER_IMAGE_NAME} bash
```

Run container
```
TFRECORD_TRAIN=/artifacts_dir/train* \
TFRECORD_EVAL=/artifacts_dir/eval* \
TFT_ARTIFACT_DIR=/artifacts_dir/ \
N_WINDOWS_TRAIN=18000 \
N_WINDOWS_EVAL=18000 \
WINDOW_SIZE=6 \
MODEL_NAME=rnn_v1 \
EPOCHS=1 \
N_AREAS=78 \
BATCH_SIZE=8 \
GPU_MEM_USAGE=0.5 \
OUTPUT_DIR=/artifacts_dir/models

python3 train.py \
--tfrecord-file-train ${TFRECORD_TRAIN} \
--tfrecord-file-eval ${TFRECORD_EVAL} \
--tft-artifacts-dir ${TFT_ARTIFACT_DIR} \
--n-windows-train ${N_WINDOWS_TRAIN} \
--n-windows-eval ${N_WINDOWS_EVAL} \
--window-size ${WINDOW_SIZE} \
--model-name ${MODEL_NAME} \
--n-areas ${N_AREAS} \
--epochs ${EPOCHS} \
--batch-size ${BATCH_SIZE} \
--output-dir ${OUTPUT_DIR} \
--gpu-memory-fraction ${GPU_MEM_USAGE}
```

### Upload container image to the GCP Conainer Registry
```
gcloud auth configure-docker --quiet

docker push ${DOCKER_IMAGE_NAME}
```
