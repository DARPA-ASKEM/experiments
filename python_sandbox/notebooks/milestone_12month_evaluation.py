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
    if re.status_code == 200:
        project = re.json()
        mkdir_write(PATH, 'project.json', project)

    # Get datasets
    re = requests.get(url = URL_DATA + f'/projects/{project_id}/assets')
    if re.status_code == 200:
        assets = re.json()

        for asset_type in assets.keys():
            print(f"{len(assets[asset_type])} {asset_type}")

    for asset_type in assets.keys():

        if asset_type in ('datasets', 'models', 'workflows', 'artifacts', 'publications'):

            for asset in tqdm(assets[asset_type]):

                # Save metadata
                asset_id = asset['id']
                mkdir_write(PATH + f'/{asset_type}/{asset_id}', f'{asset_type[:-1]}.json', asset)

                # Datasets (CSV)
                if asset_type in ('datasets', 'artifacts'):
                    for filename in asset['file_names']:
                        
                        # Get download URL 
                        re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}/download-url', params = {'filename': filename})
                        if re.status_code == 200:
                            download_url = re.json()['url']

                        # Get file
                        re = requests.get(url = download_url)
                        if re.status_code == 200:
                            data = [row.split(',') for row in re.content.decode('utf-8').split('\n')]
                            mkdir_write(PATH + f'/{asset_type}/{asset_id}', filename, data)

                # Models
                if asset_type == 'models':

                    # Get model AMR
                    filename = 'model_amr.json'
                    re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}')
                    if re.status_code == 200:
                        data = re.json()
                        mkdir_write(PATH + f'/{asset_type}/{asset_id}', filename, data)

                    # Get model configurations
                    filename = 'model_configuration.json'
                    re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}/model_configurations')
                    if re.status_code == 200:
                        for model_configuration in re.json():
                            config_id = model_configuration['id']
                            mkdir_write(PATH + f'/{asset_type}/{asset_id}/model_configurations/{config_id}', filename, model_configuration)

                # Workflows
                if asset_type == 'workflows':

                    filename = 'workflow.json'
                    re = requests.get(url = URL_DATA + f'/{asset_type}/{asset_id}')
                    if re.status_code == 200:
                        data = re.json()
                        mkdir_write(PATH + f'/{asset_type}/{asset_id}', filename, data)

                # Publications
                if asset_type == 'publications':

                    filename = 'publication.json'
                    re = requests.get(url = URL_DATA + f'/external/{asset_type}/{asset_id}')
                    if re.status_code == 200:
                        data = re.json()
                        mkdir_write(PATH + f'/{asset_type}/{asset_id}', filename, data)

# %%
# Pascale evaluation project
PATH = '../../thin-thread-examples/milestone_12month/evaluation/EVAL'
project_id = '37'
get_terarium_project(project_id)

# %%
# Sabina ensemble challenge project
PATH = '../../thin-thread-examples/milestone_12month/evaluation/ensemble_eval_SA'
project_id = '46' 
get_terarium_project(project_id)
