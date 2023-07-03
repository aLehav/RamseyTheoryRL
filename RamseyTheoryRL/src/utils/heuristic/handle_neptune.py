import neptune
import os
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.utils import stringify_unsupported

def init_neptune_model(project, name='Heuristic esimator', key='HEUR'):
    """
    Creates a neptune.Model object for a given key, project, and name.
    """
    api_token = os.environ.get('NEPTUNE_API_TOKEN')
    neptune.init_model(name=name, key=key, project=project, api_token=api_token)

def init_neptune_run(params, project):
    """
    Takes in the project and a set of params and returns
    a neptune.Run object.
    """
    api_token = os.environ.get('NEPTUNE_API_TOKEN')
    run = neptune.init_run(project=project, api_token=api_token)
    run['parameters'] = stringify_unsupported(params)
    return run
    
def init_neptune_model_version(params, project, model_name):
    """
    Takes in the project and a set of params and returns
    a neptune.ModelVersion object.
    """
    api_token = os.environ.get('NEPTUNE_API_TOKEN')
    model_version = neptune.init_model_version(model=model_name, project=project, api_token=api_token)
    model_version['parameters'] = stringify_unsupported(params)
    return model_version

def load_neptune(project, model_name, model_id, run_id):
    """
    Load in neptune Run and ModelVersion objects.
    """
    api_token = os.environ.get('NEPTUNE_API_TOKEN')
    run = neptune.init_run(with_id=run_id, project=project, api_token=api_token)
    model_version = neptune.init_model_version(with_id=model_id, model=model_name, project=project, api_token=api_token)
    return run, model_version

def get_neptune_cbk(run: neptune.Run):
    return NeptuneCallback(run=run, base_namespace="training")
