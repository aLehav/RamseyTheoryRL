import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
import neptune
import os
from neptune.integrations.tensorflow_keras import NeptuneCallback
import glob
from utils.heuristic.create_heuristic import create_model

def init_neptune_model(project,
                       name='Heuristic esimator',
                       key='HEUR'
                       ):
    """
    Creates a neptune.Model object for a given key, project, and name.
    """
    api_token = os.environ.get('NEPTUNE_API_TOKEN')
    neptune.init_model(
        name=name,
        key=key,
        project=project,
        api_token=api_token
    )

def init_neptune(params,
                 project, 
                 model_name
                 ):
    """
    Takes in the project and model names and a set of params and returns
    a neptune.Run object and a neptune.ModelVersion object.
    """
    # Retrieve the API token from the environment variable
    api_token = os.environ.get('NEPTUNE_API_TOKEN')

    run = neptune.init_run(
        project=project,
        api_token=api_token
    )

    model_version = neptune.init_model_version(
        model=model_name,
        project=project,
        api_token=api_token)

    run['parameters'] = params

    return run, model_version

def load_neptune(project,
                 model_name,
                 model_id,
                 run_id
                 ):
    """
    Load in neptune Run and ModelVersion objects.
    """
    # Retrieve the API token from the environment variable
    api_token = os.environ.get('NEPTUNE_API_TOKEN')

    run = neptune.init_run(
        with_id=run_id,
        project=project,
        api_token=api_token
    )

    model_version = neptune.init_model_version(
        with_id=model_id,
        model=model_name,
        project=project,
        api_token=api_token)

    return run, model_version

def get_neptune_cbk(run: neptune.Run):
    return NeptuneCallback(run=run, base_namespace="training")

def stop_neptune_objects(neptune_objects):
    """
    Typically to be used for Run and ModelVersion objects
    """
    for neptune_object in neptune_objects:
        neptune_object.stop()
