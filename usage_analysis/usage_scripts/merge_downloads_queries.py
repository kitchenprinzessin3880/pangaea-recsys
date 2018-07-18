import json
from itertools import chain
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath('__file__'))))
result_dir = parent_dir + '\\results'
downloads = json.load(open(result_dir+'\\downloads.json'))
queries = json.load(open(result_dir+'\\queries.json'))

print("Len Downloads, Queries :",len(downloads.keys()),len(queries.keys()))
print("Difference :",len(list(set(queries.keys()) - set(downloads.keys()))))
super_dict = {}
for k, v in chain(downloads.items(), queries.items()):
    k = int(k)
    super_dict.setdefault(k, {}).update(v)

print("Len Merge :",len(super_dict.keys()))
with open(result_dir+'\\usage_combined.json', 'w') as outfile:
    json.dump(super_dict, outfile)

