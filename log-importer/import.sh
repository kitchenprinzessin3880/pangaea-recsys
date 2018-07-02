#!/bin/sh

FILE=$1

# Elasticsearch config
SERVER=http://localhost:9200
INDEX=pangaea-recommender
TYPE=dataset

# Tools config (add your install dir, if not in path)
JQ=./jq.exe

curl -s -XDELETE "$SERVER/$INDEX"
curl -s -XPUT "$SERVER/$INDEX" --data-binary @indexconfig.json

cat $FILE | $JQ --compact-output 'to_entries[] | {"index": { "_id" : .key }}, .value' | curl -s -XPOST "$SERVER/$INDEX/$TYPE/_bulk" --data-binary @-
