# %%[markdown]
# Author: Nelson Liu
# 
# Email: [nliu@uncharted.software](mailto:nliu@uncharted.software)

# %%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Optional, Any, NoReturn
from tqdm import tqdm
import requests

# %%
API = 'https://xdd.wisc.edu/api'
MAX_NUM_PER_PAGE = 500

# %%
def build_coocc_matrix(
    api: str = API, 
    route: str = 'articles', # 'snippets'
    params: dict = {
        'dataset': 'xdd-covid-19', 
        'term': 'github',
        'dict': 'genes,covid-19_drugs', 
        'full_results': 'false', # if True, results not ranked
        'max': str(MAX_NUM_PER_PAGE),
        'include_score': 'true'
    },
    remove_empty = True,
    sort = True,
    plot: bool = True,
    plot_savepath: str = '../figures/coocc_matrix.png'
    ) -> Any:

    # Plot result
    def plot_coocc_matrix(coocc: Any, dict_terms: list, params: dict, num_top: int = 20) -> Any:

        num_rows = len(dict_terms[0])
        num_cols = len(dict_terms[1])
        m = min([num_top, num_rows])
        n = min([num_top, num_cols])

        # Labels
        labels_0 = np.array(list(dict_terms[0].keys()))
        labels_1 = np.array(list(dict_terms[1].keys()))

        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        __ = ax.imshow(np.log10(coocc[:m, :n] + 0.1), cmap = 'cividis', origin = 'upper')

        ax.set_xticks(np.arange(0, n), labels = labels_1[:n])
        ax.set_yticks(np.arange(0, m), labels = labels_0[:m])

        __ = plt.setp(ax, 
            xlabel = params['dict'].split(',')[1], 
            ylabel = params['dict'].split(',')[0],
            title = f"Co-Occurrence of Dictionary Terms in Articles (Query = {params['term'].split(',')})"
        )
        __ = plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

        for i in np.arange(m):
            for j in np.arange(n):
                __ = ax.text(j, i, f"{coocc[i, j]:d}", ha = 'center', va = 'center', color = 'w')

        return fig


    # Check if `dataset` in list of available datasets
    url = api + '/sets?all'
    r = requests.get(url)
    if r.status_code != 200:
        raise ConnectionError(f"Dataset request failed with status code {r.status_code}")
    datasets = r.json()['success']['data']
    if params['dataset'].lower() not in [d['name'].lower() for d in datasets]:
        raise KeyError(f"Given dataset value {params['dataset']} not valid")

    # Check number of dictionaries in `dict`
    if len(params['dict'].split(',')) != 2:
        raise ValueError(f"Number of given dictionaries must be 2")

    # Check if `dict` in list of available dictionaries
    url = api + '/dictionaries?all'
    r = requests.get(url)
    if r.status_code != 200:
        raise ConnectionError(f"Dictionary request failed with status code {r.status_code}")
    dicts = r.json()['success']['data']
    for d_ in params['dict'].split(','):
        if d_.lower() not in [d['name'].lower() for d in dicts]:
            raise KeyError(f"Given dictionary value {d_.lower()} not valid")
    
    # Check if `full_results` and `include_score` are boolean
    # Delete key if False
    for p in ('full_results', 'include_score'):
        if params[p].lower() not in ['true', 'false']:
            raise ValueError(f"`{p}` must be either `true` or `false`")
        elif params[p].lower() == 'false':
            del params[p]

    # Check if `max` is an int > 0
    try:
        i = int(params['max'])
        if i <= 0:
            raise ValueError
    except ValueError as err:
        print(f"`max` must be a positive integer")

    # Get list of terms of each given dictionary
    dict_terms = []
    for d_ in params['dict'].split(','):

        url = api + f"/dictionaries?dictionary={d_}&show_terms=true"
        r = requests.get(url)
        if r.status_code != 200:
            raise ConnectionError(f"Dictionary request failed with status code {r.status_code}")

        dict_terms.append({k: v for v, k in enumerate(r.json()['success']['data'][0]['term_hits'].keys())})
        print(f"Dictionary {d_} has {len(dict_terms[-1])} terms")

    # Initialize co-occurrence matrix as sparse array
    num_rows = len(dict_terms[0])
    num_cols = len(dict_terms[1])
    coocc = sp.sparse.dok_array((num_rows, num_cols), dtype = np.int32)

    # Retrieve co-occurrence data
    r = requests.get(api + f"/{route}", params = params)
    if 'hits' in r.json()['success'].keys():
        print(f"{r.json()['success']['hits']} {route} from query:\n{r.url}")
    else:
        print(f"{len(r.json()['success']['data'])} {route} from query:\n{r.url}")

    if r.status_code != 200:
        raise ConnectionError(f"Co-occurrence request failed with status code {r.status_code}")

    # Populate array
    for text in r.json()['success']['data']:
        if len(text['known_terms']) == 2:
            for i in list(text['known_terms'][0].values())[0]:
                for j in list(text['known_terms'][1].values())[0]:
                    coocc[dict_terms[0][i], dict_terms[1][j]] += 1

    # Next page if available
    if 'next_page' in r.json()['success'].keys():

        num_pages = int(np.ceil(r.json()['success']['hits'] / MAX_NUM_PER_PAGE))
        page_num = 0
        next_page = r.json()['success']['next_page']

        with tqdm(total = num_pages) as pbar:

            while next_page != '':

                # Retrieve co-occurrence data
                r = requests.get(next_page)
                if r.status_code != 200:
                    raise ConnectionError(f"Co-occurrence request failed with status code {r.status_code}")

                # Populate array
                for text in r.json()['success']['data']:
                    if len(text['known_terms']) == 2:
                        for i in list(text['known_terms'][0].values())[0]:
                            for j in list(text['known_terms'][1].values())[0]:
                                coocc[dict_terms[0][i], dict_terms[1][j]] += 1

                pbar.update(1)
                next_page = r.json()['success']['next_page']

    # Remove empty rows & columns
    if remove_empty == True:
        i = np.sum(coocc, axis = 1) != 0
        j = np.sum(coocc, axis = 0) != 0
        coocc_ = coocc[i, :]
        coocc_ = coocc_[:, j]
        coocc = coocc_
        print(f"Non-empty matrix shape: {coocc.shape[0]} x {coocc.shape[1]}")

    # Sort rows & columns
    if sort == True:
        i = np.argsort(np.sum(coocc, axis = 1))[::-1]
        coocc = coocc[i, :]
        j = np.argsort(np.sum(coocc, axis = 0))[::-1]
        coocc = coocc[:, j]

        # Re-order labels
        labels_0 = np.array(list(dict_terms[0].keys()))[i]
        labels_1 = np.array(list(dict_terms[1].keys()))[j]

        dict_terms[0] = {l: i for i, l in enumerate(labels_0)}
        dict_terms[1] = {l: i for i, l in enumerate(labels_1)}

    # Optional plotting
    if plot == True:
        fig = plot_coocc_matrix(coocc.todense(), dict_terms, params, num_top = 20)
        fig.savefig(plot_savepath, dpi = 300)

    return coocc, dict_terms, params

# %%[markdown]
## Default - Top N, Ranked

# %%
coocc, dict_terms, params = build_coocc_matrix()

# %%[markdown]
## Option - Full Results, Unranked

params = {
    'dataset': 'xdd-covid-19', 
    'term': 'github',
    'dict': 'genes,covid-19_drugs', 
    'full_results': 'true',
    'max': str(MAX_NUM_PER_PAGE),
    'include_score': 'true'
}

coocc, dict_terms, params = build_coocc_matrix(params = params)

# %%
