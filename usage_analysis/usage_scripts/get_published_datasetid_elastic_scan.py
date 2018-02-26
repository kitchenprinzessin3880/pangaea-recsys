# MIT License
#
# Copyright (c) 2017 Anusuriya Devaraju
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from elasticsearch import Elasticsearch
import logging
import pickle
import time
import datetime as dt

logging.basicConfig()
es = Elasticsearch('http://ws.pangaea.de/es',port=80)
ES_INDEX='pangaea'
DOC_TYPE= 'panmd'

start_time = time.time()
print ('Start Time: '+time.strftime("%H:%M:%S"))

#Get total datasets
#page = es.search(index = ES_INDEX,doc_type =DOC_TYPE,scroll = '1m',size = 1000,body = {"query": {"match_all": {}}, "stored_fields": [], "_source": "false"})
#scroll_size = page['hits']['total']
#print('Total datasets:',scroll_size)

# scan function in standard elasticsearch python API
rs = es.search(index=ES_INDEX, doc_type =DOC_TYPE,
               scroll='10s',
               size=1000, _source = "false",
               body={
                   "query": {"match_all": {}}
               })

data = []
sid = rs['_scroll_id']
scroll_size = rs['hits']['total']
print(scroll_size)
#before you scroll, process your current batch of hits
data = rs['hits']['hits']

while (scroll_size > 0):
    try:
        scroll_id = rs['_scroll_id']
        rs = es.scroll(scroll_id=scroll_id, scroll='60s')
        data += rs['hits']['hits']
        scroll_size = len(rs['hits']['hits'])
    except:
        break

ids =[]
for dobj in data:
    ids.append(dobj["_id"])

with open('ids.p', 'wb') as fp:
    pickle.dump(ids, fp)

secs =  (time.time() - start_time)
print('Total Execution Time: '+str(dt.timedelta(seconds=secs)))