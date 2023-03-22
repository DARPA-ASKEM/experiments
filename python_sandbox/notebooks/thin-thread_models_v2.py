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
from xml.etree import ElementTree

# %%
REST_URL_NIH_IDCONVERTER = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0'
REST_URL_XDD = 'https://xdd.wisc.edu/api'
REST_URL_BIOMODELS = 'https://www.ebi.ac.uk/biomodels'
REST_URL_MIRA = 'http://34.230.33.149:8771/api'

# %%[markdown]
# # Thin-Thread BioModels Models from MIRA (v2)
#
# Now with metadata!

# %%
# v2 model list from Ben G.
models = pd.read_csv('../../thin-thread-examples/mira_v2/mira_biomodels.tsv', sep = '\t')

# %%
# Fetch artifacts from a list of thin-thread example models
#
# 1. Document DOI
# 2. xDD identifier
# 3. Source code (SBML & additional files)
# 4. Meta-model template (MMT) representation
# 5. Petri-net representation

source = 'biomodels'
errors = []

for i, (biomodels_id, name, author, year, pmid, doi) in tqdm(models.iterrows(), total = models.shape[0]):

    # If missing DOI, try mapping from PMID using NIH ID Converter API
    if pd.isnull(doi) & ~pd.isnull(pmid):

        pmid = int(pmid)
        res = requests.get(f'{REST_URL_NIH_IDCONVERTER}/?tool=my_tool&email=my_email@example.com&ids={pmid}')
        
        if res.status_code == 200:
            root = ElementTree.fromstring(res.content)
            for child in root:
                if (child.tag == 'record') & ('doi' in child.attrib.keys()):
                    doi = child.attrib['doi']
                else:
                    doi = None
        else:
            doi = None
    elif pd.isnull(doi):
        doi = None
    else:
        pass

    # Try to find xDD ID from DOI
    if doi != None:
        res = requests.get(f'{REST_URL_XDD}/articles?doi={doi}')
        if res.status_code == 200:
            if len(res.json()['success']['data']) > 0:
                xdd_gddid = res.json()['success']['data'][0]['_gddid']
            else:
                xdd_gddid = None
        else:
            xdd_gddid = None
    else:
        xdd_gddid = None

    # Get SBML and any additional source files
    if source == 'biomodels':
        res = requests.get(f'{REST_URL_BIOMODELS}/model/files/{biomodels_id}?format=json')
        if res.status_code == 200:

            # SBML file
            main_filename = res.json()['main'][0]['name']
            additional_filenames = [f['name'] for f in res.json()['additional']]
            res = requests.get(f'{REST_URL_BIOMODELS}/model/download/{biomodels_id}?filename={main_filename}')
            if res.status_code == 200:
                model_sbml = res.content
            else:
                model_sbml = None

            # Additional files
            model_add = []
            for filename in additional_filenames:
                res = requests.get(f'{REST_URL_BIOMODELS}/model/download/{biomodels_id}?filename={filename}')
                if res.status_code == 200:
                    model_add.append(res.content)

        else:
            main_filename = None
            model_sbml = None

            additional_filenames = []
            model_add = []

    # Get MMT data from MIRA
    res = requests.get(f'{REST_URL_MIRA}/{source}/{biomodels_id}')
    if res.status_code == 200:
        model_mmt = res.json()
        model_mmt_templates = {'templates': model_mmt['templates']}
        model_mmt_parameters = {'parameters': model_mmt['parameters']}
        model_mmt_annotations = {'annotations': model_mmt['annotations']} # new in v2

        # Initial conditions
        # Find all state variables
        state_vars = [t['subject'] for t in model_mmt['templates'] if 'subject' in t.keys()]
        state_vars.extend([t['outcome'] for t in model_mmt['templates'] if 'outcome' in t.keys()])
        state_vars.extend([i for t in model_mmt['templates'] if 'controllers' in t.keys() for i in t['controllers']])
        state_vars.extend([t['controller'] for t in model_mmt['templates'] if 'controller' in t.keys()])
        state_vars_uniq = {hash(json.dumps(v, sort_keys = True, default = str, ensure_ascii = True).encode()): v for v in state_vars}
        model_mmt_initials = {'initials': {var['name']: {**var, **{'value': None}} for var in state_vars_uniq.values()}}

        # Populate with given values
        for k, v in model_mmt['initials'].items():
            if k in model_mmt_initials['initials'].keys():
                model_mmt_initials['initials'][k]['value'] = v

        # Get Petri net
        res = requests.post(f'{REST_URL_MIRA}/to_petrinet', json = model_mmt)
        if res.status_code == 200:
            model_petri = res.json()
        else:
            model_petri = None
    else:
        model_mmt = None
        model_mmt_templates = None
        model_mmt_parameters = None
        model_mmt_initials = None
        model_mmt_annotations = None
        model_petri = None

    # Create artifact directory if not exist
    path = f'../../thin-thread-examples/mira_v2/{source}/{biomodels_id}'
    if os.path.exists(path) == False:
        os.makedirs(path)


    # Write artifact files
    for data, filename in zip([doi, xdd_gddid, model_sbml, model_mmt, model_mmt_templates, model_mmt_parameters, model_mmt_initials, model_mmt_annotations, model_petri], ['document_doi.txt', 'document_xdd_gddid.txt', main_filename, 'model_mmt.json', 'model_mmt_templates.json', 'model_mmt_parameters.json', 'model_mmt_initials.json', 'model_mmt_annotations.json', 'model_petri.json']):

        if data != None:
            
            # SBML XML file
            if filename.split('.')[-1] == 'xml':

                # `src` directory
                if os.path.exists(path + '/src') == False:
                    os.mkdir(path + '/src')
                
                # `src/main` directory
                if os.path.exists(path + '/src/main') == False:
                    os.mkdir(path + '/src/main')
                
                with open(path + f'/src/main/{filename}', 'wb') as f:
                    f.write(data)

            else:
                with open(path + f'/{filename}', 'w') as f:
                    if isinstance(data, dict):
                        f.write(json.dumps(data, indent = 4))
                    else:
                        f.write(data)

            # models.at[i, filename] = True

        else:
            msg = f'Error: {biomodels_id} {filename} data = None'
            errors.append(msg)

            models.at[i, filename] = False

    # Write any additional source files
    for data, filename in zip(model_add, additional_filenames):

        # `src/additional` directory
        if os.path.exists(path + '/src/additional') == False:
            os.mkdir(path + '/src/additional')

        with open(path + f'/src/additional/{filename}', 'wb') as f:
            f.write(data)

# Print error messages
__ = [print(msg) for msg in errors]

# %%
models.to_csv('../../thin-thread-examples/mira_v2/mira_biomodels_.tsv', sep = '\t')

# %%
