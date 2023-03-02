
This document provides an overview on how to prepare and use Elasticsearch for semantic search in your application.
Even though we will mention optimizations aspects for Elasticseach 8, we will only provide a full examples for non-optimized queries.

# Requirements

1. Elasticsearch version 7.3 or highter. For indexes of more than a million documents, Elasticsearch 8 recommended.
2. An script, library, or program which will use a large-language-model to generate text embeddings. The file `./search.py` contains an example, which we'll go over below.


# Required Index Mappings

In order to use semantic search for a given Elasticsearch index, you'll need to add a new property to its index mappings.
This new field will be of type `dense_vector`, which we will populate later on with the contents embeddings- an array of floating numbers.

As an example, let's say we need to support semantic search through paragraphs, on our `paragraphs` index. We can add a property with a custom name such as `vector` or `embedding`. Let's use `embeddings` here:

```json
"paragraphs": {
  "mappings": {
    "properties": {
      "paragraph_text": {
         ...
      },
      ...other_properties,

      "embeddings": {
        "type": "dense_vector",
        "dims": 768,
        "index": false    # false or absent for Elasticsearch 7
                          # true for Elasticsearch 8. See last section in this document.
      }

    }
  }
}
```

The vector dimension, or `dim`, is the length of the embeddings vector (array of floats). This should be consistent in length across documents- accomplished by using the same algorithm and model to create the text embeddings. The `index` subproperty in our `embeddings` example  is set to false, but needs to be set to `true` in order to use search optimizations supported only in Elasticsearch 8\*.

# Generating Embeddings

The file `./search.py` includes an `MPNetEmbedder` class, with an `embed` method. It accepts a list of strings, and outputs a list of embeddings. The `embed` method can be customized, but for our purposes let's embed a sentence using the existing code as-is:

```python
model_engine = MPNetEmbedder()

output_embeddings = model_engine.embed(["I am an example sentence, to be embedded."])

# As this particular `embed()` function supports a list as input, it outputs a list as well
# Each output index corresponds to the embeddings of the input sentence on the same index
my_sentence_embedding = output_embeddings[0]

```

# Populating the dense_vector field for each Document

Populating the new `dense_vector` field for indexes with previous data usually consists of using a program to bulk update each document and populate the embeddings using a function similar to `MPNetEmbedder.embed()`. Whenever a new document is created or subsequently updated, if the input data for the embedder changes, the embeddings (dense_vector) field will need to be updated.

# Querying Documents using Semantic Search

Now that the index has a dense_vector `embeddings` property, and the documents within it all have this field populated, we can use input text to perform a semantic search.

First, a sketch of how the query will look like with some hardcoded input in Kibana or HTTP:

```
POST /paragraphs/_search
{
 "script_score": {
   "query": {"match_all": {}},
   "script": {
     "source": "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
     "params": {"query_vector": [0.3, 0.4, 0.002, -0.203, ..., 0.333]}
   }
 }
}
```

Note that the query vector is hardcoded. We expect the MPNet embedder to create vectors with 768 items, so we have redacted it in our example. For non-optimized search, this will scan an index (match_all) and return all the documents sorted by highest match first.


# Putting It All Together

A full example on embedding an input query and performing a search follows:

```python

# pip install elasticsearch
from elasticsearch import Elasticsearch
from search import MPNetEmbedder

es = Elasticsearch()
model_engine = MPNetEmbedder()

# Sample input query to be embedded, and compared to elasticsearch documents
input_query = "Pellentesque tristique imperdiet tortor."

query_embedding = model_engine.embed([input_query])[0]

es_query = {
    "query": {
      "script_score": {
        "query": {"match_all": {}},
        "script": {
          # ES does not allow negative scores. We can either:
          # - a) clamp at 0, and not allow negatives, or
          # - b) Add 1 to the result to compare score
          # Here we use option (a):
          "source": "Math.max(cosineSimilarity(params.query_vector, 'embeddings'), 0)",
          "params": {
            "query_vector": query_embedding
          }
        }
      }
    },
    "_source": {
      # exclude embeddings in document results
      "excludes": ["embeddings"]
    }
  }

results = es.search(index="paragraphs", body=es_query, scroll="2m", size=100)

```

## Elasticsearch 8

\* Other settings can be enabled to perform faster semantic searches.
Index mappings and query changes might be needed. See the below documentation on Approximate kNN search and the dense_vector field type, as well as filtering input and output documents:

https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html

https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html
