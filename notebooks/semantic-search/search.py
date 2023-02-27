"""
pip install pandas
pip install torch
pip install transformers
pip install easyrepl


download dataset from https://www.kaggle.com/datasets/benhamner/nips-papers
extract into data/ folder (mainly just want papers.csv)
"""
from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd


import pdb



def get_data(rows=400):
    #read in data, and remove rows with missing abstracts
    df = pd.read_csv('data/papers.csv')
    df = df[['title', 'abstract', 'paper_text']]
    df = df[df['abstract'] != 'Abstract Missing']
    df.reset_index(inplace=True, drop=True)

    #take a random subset of the data since it all probably won't fit in memory
    df = df.sample(rows, random_state=42)
    df.reset_index(inplace=True, drop=True)

    return df


def get_paragraph_data():
    df = get_data()
    df['paragraphs'] = df['paper_text'].apply(lambda x: x.split('\n'))

    pdb.set_trace()

    return df

class SpecterEmbedder:
    def __init__(self, try_cuda=True):
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter')

        # move model to GPU if available
        if try_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

        # store whether on GPU or CPU
        self.device = next(self.model.parameters()).device

    def embed(self, texts:list[str]):
        # important to use no_grad otherwise it uses way too much memory
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            result = self.model(**inputs)

            embeddings = result.last_hidden_state[:, 0, :]

            return embeddings


class BertEmbedder:
    def __init__(self, try_cuda=True):
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

        # move model to GPU if available
        if try_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()

        # store whether on GPU or CPU
        self.device = next(self.model.parameters()).device

    def embed(self, texts:list[str]):
        # important to use no_grad otherwise it uses way too much memory
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            result = self.model(**inputs)

            embeddings = result.last_hidden_state[:, 0, :]

            return embeddings



def main():
    from easyrepl import REPL

    embedder = BertEmbedder()#SpecterEmbedder()

    #embed the corpus of data. Treat title+abstract as one document
    df = get_data()
    title_abs = (df['title'] + embedder.tokenizer.sep_token + df['abstract']).tolist()
    corpus_embeddings = embedder.embed(title_abs)


    #number of results to display
    top_k = 5

    print('Enter a query to search the corpus. ctrl+d to exit')
    for query in REPL():
        query_embedding = embedder.embed([query])

        # compute similarity scores
        scores = torch.nn.functional.cosine_similarity(query_embedding, corpus_embeddings, dim=1)
        
        #get the indices of the top_k scores
        top_results = torch.topk(scores, k=top_k).indices.tolist()

        #print the results
        for idx in top_results:
            print(f"Title: {df.loc[idx, 'title']}")
            print(f"Abstract: {df.loc[idx, 'abstract']}")
            print(f"Score: {scores[idx]}")
            print()

        print('-------------------------------------------------')
        


if __name__ == '__main__':
    main()