import configparser as ConfigParser
import os
import argparse
import re
import pandas as pd
import time
import datetime as dt
from scipy import sparse
import numpy as np
import tables
import pickle
import json
from scipy.sparse import coo_matrix,csr_matrix
from pandas.api.types import CategoricalDtype

class ProcessLogs:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--config", required=True,help="Path to usage.ini config file")
        args = ap.parse_args()
        config = ConfigParser.ConfigParser()
        config.read(args.config)
        self.source_folder = config['DATASOURCE']['source_folder']
        self.source_file_prefix = config['DATASOURCE']['source_file_prefix']
        self.source_file_suffix = config['DATASOURCE']['source_file_suffix']
        self.num_top_dataset = int(config['DATASOURCE']['number_of_reldatasets'])
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath('__file__'))))
        self.source_dir = os.path.abspath(os.path.join(parent_dir, self.source_folder))
        self.hdf_file = config['DATASOURCE']['hdf_file_name']

        # read file with data ids
        self.published_datasets = []
        idfile_dir = parent_dir + '\\usage_scripts\\results\\ids.p'
        with open(idfile_dir, 'rb') as fp:
            self.published_datasets = pickle.load(fp)

    def parse_str(self,x):
        #remove double quotes
        if x:
            return x[1:-1]
        else:
            return x

    def cleanLogs(self, dfmain):
        # Filter out non GET and non 200 requests
        request = dfmain.request.str.split()
        dfmain = dfmain[(request.str[0] == 'GET') & (dfmain.status == 200)]
        #unwanted resources
        dfmain = dfmain[~dfmain['request'].str.match(r'^/media|^/static|^/admin|^/robots.txt$|^/favicon.ico$')]
        # filter crawlers by User-Agent
        dfmain = dfmain[~dfmain['user_agent'].str.match(r'.*?bot|.*?spider|.*?crawler|.*?slurp', flags=re.I).fillna(False)]
        download_indicators = ['format=textfile', 'format=html', 'format=zip']
        download_joins = '|'.join(map(re.escape, download_indicators))
        dfmain = dfmain[(dfmain.request.str.contains(download_joins))]
        # get request uri, extract data id
        dfmain['_id'] = dfmain['request'].str.extract(r'PANGAEA.\s*([^\n? ]+)', expand=False)
        # remove rows if dataset is NaN
        dfmain = dfmain.dropna(subset=['_id'], how='all')
        # handle escaped char (string to int) PANGAEA.56546%20,PANGAEA.840721%0B
        dfmain._id = dfmain._id.apply(lambda x: x.split('%')[0])
        dfmain._id = dfmain._id.astype(int)
        # convert time
        #dfmain['time'] = dfmain['time'].str.strip('[]').str[:-6]
        #dfmain['time'] = pd.to_datetime(dfmain['time'], format='%d/%b/%Y:%H:%M:%S')
        #dfmain['time'] = dfmain['time'].dt.date
        return dfmain

    def readLogs(self):
        dfs = []
        for file in os.listdir(self.source_dir):
            if file.startswith(self.source_file_prefix) and file.endswith(self.source_file_suffix):
                filepath = os.path.join(self.source_dir, file)
                data = pd.read_csv(filepath, compression='bz2', encoding='ISO-8859-1',
                                   sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', engine='python', header=0,
                                   usecols=[0, 3, 4, 5, 8],
                                   names=['ip','time','request', 'status', 'user_agent'],
                                   converters={"request": self.parse_str})
                df = self.cleanLogs(data)
                dfs.append(df)

        # Concatenate all data into one DataFrame
        df_final = pd.concat(dfs, ignore_index=True)
        print("before",str(df_final.shape))
        # exlude rows that contains old data
        df_final = df_final[df_final['_id'].isin(self.published_datasets)]
        print("after",str(df_final.shape))
        return df_final


    def store_sparse_mat(self,M, name, filename):
        """
        Store a csr matrix in HDF5
        Parameters
        ----------
        M : scipy.sparse.csr.csr_matrix
            sparse matrix to be stored
        name: str
            node prefix in HDF5 hierarchy
        filename: str
            HDF5 filename
        """
        assert (M.__class__ == sparse.csr.csr_matrix), 'M must be a csr matrix'
        with tables.open_file(filename, 'a') as f:
            for attribute in ('data', 'indices', 'indptr', 'shape'):
                full_name = f'{name}_{attribute}'

                # remove existing nodes
                try:
                    n = getattr(f.root, full_name)
                    n._f_remove()
                except AttributeError:
                    pass

                # add nodes
                arr = np.array(getattr(M, attribute))
                atom = tables.Atom.from_dtype(arr.dtype)
                ds = f.create_carray(f.root, full_name, atom, arr.shape)
                ds[:] = arr

    def load_sparse_mat(self,name, filename='store.h5'):
        """
        Load a csr matrix from HDF5
        Parameters
        ----------
        name: str
            node prefix in HDF5 hierarchy
        filename: str
            HDF5 filename
        Returns
        ----------
        M : scipy.sparse.csr.csr_matrix
            loaded sparse matrix
        """
        with tables.open_file(filename) as f:
            # get nodes
            attributes = []
            for attribute in ('data', 'indices', 'indptr', 'shape'):
                attributes.append(getattr(f.root, f'{name}_{attribute}').read())

        # construct sparse matrix
        M = sparse.csr_matrix(tuple(attributes[:3]), shape=attributes[3])
        return M

    #e following function uses a generator factory to and uses a double for loop over a cartesian product of chunks.
    # We precalculate the norms for each 't' ndarray.
    #we extract the cosine similarities by calculating them
    #ourselves over each sub-chunk and track where the similarities were greater than our threshold.
    def f(self, t, c, p=-1, v=False):
        n = (t ** 2).sum(1) ** .5 #RuntimeWarning: invalid value encountered in power
        g = lambda: ((x, t[x:x + c]) for x in range(0, t.shape[0], c))
        h = lambda a, b, i, j: a.dot(b.T) / n[i:i + c, None] / n[j:j + c]
        d = lambda s: (s * (1 - np.eye(s.shape[0])))

        for i, a in g():
            for j, b in g():
                s = h(a, b, i, j)
                if i == j:
                    s = d(s)
                i_, j_ = np.where(s > p)
                if v:
                    print('\r', 'i = {:0000000d}; j = {:0000000d}'.format(i, j), end='')
                yield s[i_, j_], i_ + i, j_ + j

    def get_Total_Related_Downloads(self,dfmain):
        #total downloads
        download_count = dfmain.groupby(['_id'])['_id'].agg(['count'])

        #build datasets vs ip similarity matrix
        group = pd.DataFrame({'download_count': dfmain.groupby(['_id', 'ip']).size()}).reset_index()
        person_u = list(group.ip.unique())
        dataset_u = list(group._id.unique())

        outF = open("results/data_list.txt", "w")
        for line in dataset_u:
            outF.write(str(line))
            outF.write("\n")
        outF.close()

        data = group['download_count'].tolist()
        #row_type = CategoricalDtype(categories=dataset_u)
        #row = group._id.astype(row_type)
        row = group._id.astype('category', categories=dataset_u).cat.codes
        col = group.ip.astype('category', categories=person_u).cat.codes
        len_dataset = len(dataset_u)
        len_person =  len(person_u)
        print("Datasets vs Ips :",str(len_dataset), str(len_person))#(309235, 81566)
        df_sparse = sparse.csr_matrix((data, (row, col)), dtype=np.int8,shape=(len_dataset, len_person))
        print('Sparse matrix size in bytes:',df_sparse.data.nbytes)
        values, *ij = zip(*self.f(df_sparse.toarray().astype(int), 100, -1, v=True))
        print(" concatening...")
        values = np.concatenate(values) # MemoryError - big data!!!!

        ij = list(map(np.concatenate, ij))
        csrmatrix = csr_matrix((values, ij))
        csrmatrix.setdiag(1)
        self.store_sparse_mat(csrmatrix, 'csrmatrix_sim', self.hdf_file)
        sim = self.load_sparse_mat('csrmatrix_sim', self.hdf_file)

        print('Sim Done!')
        m, n = sim.shape
        print('Sim size in bytes:' ,sim.data.nbytes)

        json_data = {}
        for i in range(m):
            target_id = int(dataset_u[i])
            simvalue = sim[i].tocoo().data
            #get indices of sorted array(decreasing)
            indices = sorted(range(len(simvalue)), key=lambda k: simvalue[k], reverse=True)
            #get dataids based on indices
            rel_datasets = [dataset_u[i] for i in indices]
            rel_datasets.remove(target_id)
            # select top-k related datasets
            rel_datasets = rel_datasets[:self.num_top_dataset]
            downloads = download_count.loc[target_id]['count']
            data = {}
            rel_datasets = [int(i) for i in rel_datasets]
            data['related_datasets'] = rel_datasets
            data['total_downloads'] = int(downloads)
            json_data[target_id] = data

        with open("results/downloads.json", 'w') as fp:
            json.dump(json_data, fp)

        #sim.stop()

start_time = time.time()
print ('Start Time: '+time.strftime("%H:%M:%S"))
c1 = ProcessLogs()
dfdwl = c1.readLogs()
#get total downloads and sim by download profiles
dfdwl = dfdwl.drop_duplicates(['time', 'ip', '_id'])
dfdwl = dfdwl[['ip', '_id']]
c1.get_Total_Related_Downloads(dfdwl)

secs =  (time.time() - start_time)
print('Total Execution Time: '+str(dt.timedelta(seconds=secs)))
