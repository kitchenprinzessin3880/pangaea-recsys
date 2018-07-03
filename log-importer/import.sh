#!/bin/bash

FILE=$1

# Elasticsearch config
SERVER=http://localhost:9200
INDEX=pangaea-recommender
TYPE=rec

set +e

echo "Deleting old index (ignore errors)..."
curl -s -XDELETE "$SERVER/$INDEX"
echo " -- DONE"

set -e

echo "Recreate index..."
curl -s -XPUT "$SERVER/$INDEX" --data-binary @indexconfig.json
echo " -- DONE"

echo "Importing recommender data..."
cat $FILE | jq --compact-output 'to_entries[] | {"index": { "_id" : .key }}, .value' \
  | split --lines=20000 --numeric-suffixes --filter="echo -n 'Importing '\$FILE'...'; curl -s -XPOST '$SERVER/$INDEX/$TYPE/_bulk' --data-binary @- >/dev/null; echo ' - DONE'" - 'chunk'
echo "IMPORT FINSIHED."
