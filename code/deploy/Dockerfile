FROM ubuntu:16.04

RUN apt-get update -y

RUN apt-get install --no-install-recommends -y -q ca-certificates python-dev python-setuptools \
                                                  wget unzip git

RUN easy_install pip

RUN pip install tensorflow==1.10
RUN pip install pyyaml==3.12 six==1.11.0

RUN wget -nv https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.zip && \
    unzip -qq google-cloud-sdk.zip -d tools && \
    rm google-cloud-sdk.zip && \
    tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    tools/google-cloud-sdk/bin/gcloud -q components update \
        gcloud core gsutil && \
    tools/google-cloud-sdk/bin/gcloud config set component_manager/disable_update_check true && \
    touch /tools/google-cloud-sdk/lib/third_party/google.py

# ADD build /ml

ENV PATH $PATH:/tools/node/bin:/tools/google-cloud-sdk/bin

WORKDIR /app

COPY service_account.json /assets/service_account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/assets/service_account.json

COPY src /app