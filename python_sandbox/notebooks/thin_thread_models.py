# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
import pandas as pd
import json
import requests
import os
from tqdm import tqdm

# %%
# Fetch artifacts from a list of thin-thread example models
#
# 1. Document DOI
# 2. xDD identifier
# 3. Meta-model template (MMT) representation
# 4. Petri net representation

# %%
models = pd.read_csv('../../thin-thread-examples/models.csv')

# %%
for i, (doi, source, info) in tqdm(models.iterrows()):

    # Get xDD ID
    res = requests.get(f'https://xdd.wisc.edu/api/articles?doi={doi}')
    if res.status_code == 200:
        xdd_gddid = res.json()['success']['data'][0]['_gddid']
    else:
        xdd_gddid = None

    # Get SBML file
    if source == 'biomodels':
        res = requests.get(f'https://www.ebi.ac.uk/biomodels/model/files/{info}?format=json')
        if res.status_code == 200:
            main_filename = res.json()['main'][0]['name']

            res = requests.get(f'https://www.ebi.ac.uk/biomodels/model/download/{info}?filename={main_filename}')
            if res.status_code == 200:
                model_sbml = res.content
            else:
                model_sbml = None
        else:
            main_filename = None
            model_sbml = None


    # Get MMT
    res = requests.get(f'http://34.230.33.149:8771/api/{source}/{info}')
    if res.status_code == 200:
        model_mmt = res.json()
        model_mmt_templates = {'templates': model_mmt['templates']}
        model_mmt_parameters = {'parameters': model_mmt['parameters']}

        # Get Petri net
        res = requests.post('http://34.230.33.149:8771/api/to_petrinet', json = model_mmt)
        if res.status_code == 200:
            model_petri = res.json()
        else:
            model_petri = None
    else:
        model_mmt = None
        model_mmt_templates = None
        model_mmt_parameters = None
        model_petri = None

    # Create artifact directory if not exist
    path = f'../../thin-thread-examples/{source}/{info}'
    if os.path.exists(path) == False:
        os.mkdir(path)

    # Write artifact files
    for data, filename in zip([doi, xdd_gddid, model_sbml, model_mmt_templates, model_mmt_parameters, model_petri], ['document_doi.txt', 'document_xdd_gddid.txt', 'model_sbml.xml', 'model_mmt_templates.json', 'model_mmt_parameters.json', 'model_petri.json']):

        if data != None:

            if filename.split('.')[-1] == 'xml':
                with open(path + f'/{filename}', 'wb') as f:
                    f.write(data)
            else:
                with open(path + f'/{filename}', 'w') as f:
                    if isinstance(data, dict):
                        f.write(json.dumps(data, indent = 4))
                    else:
                        f.write(data)

        else:
            print(f'Error: {info} {filename} data = None')


# %%





