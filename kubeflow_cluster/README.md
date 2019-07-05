# KubeFlow setup for Chicago Taxi Trips forecast

This is a mini tutorial that comprises the usage of [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/) for time-series forecast problem. For more details, please read the [official documentation](https://www.kubeflow.org/docs/gke/deploy/deploy-cli/).


## How to create a KubeFlow cluster on GCP

Install the [Cloud SDK](https://cloud.google.com/sdk/docs/)

Install the Kubernetes command-line tool

```
gcloud components install kubectl
```

Install the Kubeflow command-line tool

Download `kfctl` [lastest version](https://github.com/kubeflow/kubeflow/releases/)

```
wget https://github.com/kubeflow/kubeflow/releases/download/v0.5.1/kfctl_v0.5.1_linux.tar.gz
tar -xvf kfctl_v0.5.1_linux.tar.gz.tar.gz
mv kfctl /usr/local/bin
```

```
#Set application default credentials
gcloud auth application-default login

# Set up OAuth credentials and Cloud IAP. This allows you to securely connect to Kubeflow web applications. To setup OAuth with IAP please read https://www.kubeflow.org/docs/gke/deploy/oauth-setup/

ROOT_DIR=${PWD}

export REGION=us-central1
export ZONE=${REGION}-a 

export DEPLOYMENT_NAME=chicago-taxi-trips
export PROJECT=ciandt-cognitive-sandbox
export KFAPP=${ROOT_DIR}/${DEPLOYMENT_NAME}

# Default uses Cloud IAP:
#OAuth 2.0 client ID: "Kubeflow Sandbox Web Client"
export CLIENT_ID=1019062845561-e36hmq1gn7eusjbp5scosqd3ava7mg4v.apps.googleusercontent.com
export CLIENT_SECRET=FXIBeBhre_eKwuc13P2ywgEu

kfctl init ${KFAPP} --platform gcp --project ${PROJECT}


# Alternatively, use this command if you want to use basic authentication:
export KUBEFLOW_USERNAME=admin
export KUBEFLOW_PASSWORD=admin
kfctl init ${KFAPP} --platform gcp --project ${PROJECT} --use_basic_auth -V

cd ${KFAPP}

# Create configuration files
kfctl generate all -V --zone ${ZONE}

# Create/Update cluster specs
kfctl apply all -V

export BUCKET_NAME=${PROJECT}-${DEPLOYMENT_NAME}-bucket
gsutil mb -c regional -l ${REGION} gs://${BUCKET_NAME}
```

Access Kubeflow web UI at https://<DEPLOYMENT_NAME>.endpoints.<PROJECT>.cloud.goog/_gcp_gatekeeper/authenticate. 
It may take a couple of minutes to the URI to be available

## Build pipeline

First, [setup up python environment](https://www.kubeflow.org/docs/pipelines/tutorials/pipelines-tutorial/#set-up-python) using miniconda ou virtualenv

```
conda create --name mlpipeline python=3.7
source activate mlpipeline
pip install -r requirements.txt --upgrade
```

Afterwards, compile the Kubeflow pipeline

```
cd code/pipeline
python pipeline.py
```

Then [run](https://www.kubeflow.org/docs/pipelines/tutorials/pipelines-tutorial/#run-the-pipeline) the pipeline using the Kubeflow Pipelines Web Interface