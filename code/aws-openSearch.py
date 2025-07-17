from opensearchpy import OpenSearch
import boto3, json

# 1. Connect (signing with SigV4 is typical in prod)
client = OpenSearch(
    hosts=[{"host": "search-my-domain.us-west-2.es.amazonaws.com", "port": 443}],
    http_auth=("admin", "my-password"),   # or AWS4Auth
    use_ssl=True, verify_certs=True
)

# 2. Hybrid query: BM25 keywords + k-NN vector
user_q = "ice age lakes"
query_emb = bedrock_embed(user_q)         # call Bedrock Titan or Cohere to embed

resp = client.search(
    index="rag-index",
    body={
      "size": 5,
      "query": {
        "bool": {
          "should": [
            {"match": {"chunk": user_q}},          # lexical
            {"knn":  {"text_vector": {"vector": query_emb, "k": 50}}}  # vector
          ]
        }
      },
      "rescore": {                                # optional boost like TagScoring
        "window_size": 50,
        "query": {
          "rescore_query": {
            "function_score": {
              "query": {"match": {"locations": "Canada"}},
              "boost": 5.0
            }
          }
        }
      }
    }
)

for hit in resp["hits"]["hits"]:
    print(hit["_score"], hit["_source"]["title"], hit["_source"]["locations"])