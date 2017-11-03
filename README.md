# pangaea-recsys

Example Query:

```json
POST /pangaea-recommender/dataset/_search HTTP/1.1
{
  "size": 5,
  "query": {
    "more_like_this": {
      "fields": [
        "query_1"
      ],
      "like": [
        {
          "_id": 837739
        }
      ],
      "min_term_freq": 0,
      "min_doc_freq": 5,
      "max_query_terms": 25,
      "minimum_should_match": "10%",
      "boost_terms": 1
    }
  }
}
```