import json
from itertools import chain

downloads = json.load(open('results/downloads.json'))
queries = json.load(open('results/queries.json'))

print("Len Downloads, Queries :",len(downloads.keys()),len(queries.keys()))
print("Difference :",len(list(set(queries.keys()) - set(downloads.keys()))))
super_dict = {}
for k, v in chain(downloads.items(), queries.items()):
    k = int(k)
    super_dict.setdefault(k, {}).update(v)

print("Len Merge :",len(super_dict.keys()))
with open('results/usage.json', 'w') as outfile:
    json.dump(super_dict, outfile)