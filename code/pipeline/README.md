# Build Kubeflow pipeline

First, [setup up python environment](https://www.kubeflow.org/docs/pipelines/tutorials/pipelines-tutorial/#set-up-python)

```
conda create --name mlpipeline python=3.7
source activate mlpipeline
pip install -r requirements.txt --upgrade
```

Afterwards, compile pipeline

```
python build_pipeline.py
```

Upload pipeline throught Web interface or running the following

```
python upload_pipeline.py
```

Then [run](https://www.kubeflow.org/docs/pipelines/tutorials/pipelines-tutorial/#run-the-pipeline) the pipeline using the Kubeflow Pipelines Web Interface