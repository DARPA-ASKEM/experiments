
This document provides an overview on how to prepare and use Elasticsearch for semantic similarity search.

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
        "index": false           # false for Elasticsearch 7, true for Elasticsearch 8 if using performant kNN search
      }

    }
  }
}
```

The vector dimention, or `dim`, is the length of the embeddings vector (array of floats). This should be consistent in length across documents- accomplished by using the same algorithm and model to create the text embeddings. The `index` subproperty in our `embeddings` example  is set to false, but needs to be set to `true` in order to use search optimizations supported only in Elasticsearch 8.

# Generating Embeddings

The file search.py includes an `MPNetEmbedder` class, with an `embed` method. It accepts a list of strings, and outputs an array of embeddings. The `embed` method can be customized to suit the application, but for our purposes let's embed a sentence using the existing code as-is:

```python
from search import MPNetEmbedder

model_engine = MPNetEmbedder()

output_embeddings = model_engine.embed(["I am an example sentence, to be embedded."])

# As this particular `embed()` function supports a list as input, it outputs a list as well
# Each output index corresponds to the embeddings of the input sentence on the same index
my_sentence_embedding = output_embeddings[0]

```

# Populating the new property of each Elasticsearch Document

Populating the new `dense_vector` field for indexes with previous data usually consists of using a program to bulk update each document and populate the embeddings using a function similar to `MPNetEmbedder.embed()`. Whenever a new document is created or subsequently updated, if the input data for the embedder changes, the embeddings property will need to be updated.

# Querying Documents using Semantic Search

Now that the index has a dense_vector `embeddings` property, and the documents within it all have this field populated, we can use input text to perform a semantic search.

## Elasticsearch 7

First, a sketch of how the query will look like with some hardcoded input in Kibana:

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

Note that the query vector is hardcoded. We expect the MPNet embedder to create vectors with 768 items, so we have redacted it in our example. On Elasticsearch 7, this will scan an index (match_all) and return all the documents sorted by highest match first.

## Elasticsearch 8

On Elasticsearch 8, ... TODO

# Putting It All Together

TODO Sample of embedding an input query and using Elasticsearch library to search
...

