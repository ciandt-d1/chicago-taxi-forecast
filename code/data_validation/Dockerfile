FROM tensorflow/tensorflow:1.13.1-py3

RUN apt-get update && apt-get install -y    

RUN mkdir /assets
COPY requirements.txt /assets
RUN pip install -r /assets/requirements.txt

COPY src /app

WORKDIR /app