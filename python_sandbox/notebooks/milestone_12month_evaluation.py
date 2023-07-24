# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%[markdown]
# 12-Month Milestone Ensemble Challenge

# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import requests
import json
import csv
import pathlib
from typing import NoReturn, Optional, Any
from tqdm import tqdm

# %%
URL_DATA = 'http://data-service.staging.terarium.ai'
URL_SIM = 'http://simulation-service.staging.terarium.ai'
PATH = '../../thin-thread-examples/milestone_12month/evaluation/ensemble_eval_SA'

project_id = '46'

# %%
def mkdir_write(path: str, filename: str, data: Any) -> NoReturn:

    # Check if path exists, else mkdir
    p = pathlib.Path(path)
    p.mkdir(parents = True, exist_ok = True)

    # Write file
    p = pathlib.Path(path + '/' + filename)
    ex = filename.split('.')[-1]
    with p.open('w') as f:
        if ex == 'json':
            f.write(json.dumps(data, indent = 2))
        elif ex == 'csv':
            w = csv.writer(f, delimiter = ',')
            w.writerows(data)
        else:
            f.write(data)

# %%
def get_terarium_project(project_id: str) -> NoReturn:

    # Get project metadata
    re = requests.get(url = URL_DATA + f'/projects/{project_id}')
    project = re.json()
    mkdir_write(PATH, 'project.json', project)

    # Get datasets
    re = requests.get(url = URL_DATA + f'/projects/{project_id}/assets')
    assets = re.json()
    print(f'{len(assets["datasets"])} datasets \n{len(assets["models"])} models')

    for asset_type in ['datasets', 'models']:

        for asset in tqdm(assets[asset_type]):

            # Save metadata
            asset_id = asset['id']
            mkdir_write(PATH + f'/{asset_type}/{asset_id}', f'{asset_type[:-1]}.json', asset)

            # Datasets (CSV)
            if asset_type == 'datasets':
                for filename in asset['file_names']:

                    # Get download URL 
                    re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}/download-url', params = {'filename': filename})
                    download_url = re.json()['url']

                    # Get file
                    re = requests.get(url = download_url)
                    data = [row.split(',') for row in re.content.decode('utf-8').split('\n')]
                    mkdir_write(PATH + f'/{asset_type}/{asset_id}', filename, data)
            

            # Models
            if asset_type == 'models':

                # Get model AMR
                filename = 'model_amr.json'
                re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}')
                data = re.json()
                mkdir_write(PATH + f'/{asset_type}/{asset_id}', filename, data)

                # Get model configurations
                re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}/model_configurations')
                filename = 'model_configuration.json'
                for model_configuration in re.json():
                    config_id = model_configuration['id']
                    mkdir_write(PATH + f'/{asset_type}/{asset_id}/model_configurations/{config_id}', filename, model_configuration)


# %%