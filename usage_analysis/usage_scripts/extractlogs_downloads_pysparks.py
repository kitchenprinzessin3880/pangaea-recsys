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
import findspark
import pyspark
from pyspark.sql.types import Row
import pyspark.sql.functions as psf
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
import pyspark.sql.functions
from pyspark import SparkConf,SparkContext,SQLContext
from pyspark.sql.types import *

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
        dfmain['time'] = dfmain['time'].str.strip('[]').str[:-6]
        dfmain['time'] = pd.to_datetime(dfmain['time'], format='%d/%b/%Y:%H:%M:%S')
        dfmain['time'] = dfmain['time'].dt.date
        return dfmain

    def readLogs(self):
        dfs = []
        for file in os.listdir(self.source_dir):
            if file.startswith(self.source_file_prefix) and file.endswith(self.source_file_suffix):
                filepath = os.path.join(self.source_dir, file)
                data = pd.read_csv(filepath, compression='bz2', encoding='ISO-8859-1',
                                   sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', engine='python', header=0,
                                   usecols=[0, 3, 4, 5, 8],
                                   names=['ip', 'time', 'request', 'status', 'user_agent'],
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
        row = group._id.astype('category', categories=dataset_u).cat.codes
        cols = group.ip.astype('category', categories=person_u).cat.codes
        len_dataset = len(dataset_u)
        len_person =  len(person_u)
        print("Datasets vs Ips :",str(len_dataset), str(len_person))#(309235, 81566)
        sparsemat = sparse.csr_matrix((data, (row, cols)), dtype=np.int8,shape=(len_dataset, len_person))
        m, n = sparsemat.shape

        def f(x):
            d = {}
            for i in range(len(x)):
                d[str(i)] = float(x[i])
            return d

        # load PySpark using findSpark package

        SparkContext.setSystemProperty('spark.executor.memory', '5g')
        SparkContext.setSystemProperty('spark.driver.memory', '5g')
        #SparkContext.setSystemProperty('spark.executor.heartbeatInterval', '1000000000s')
        findspark.init()
        #conf = SparkConf().setAppName("simdownload")
        #conf = (conf.setMaster('local[*]').set('spark.executor.memory', '4G'))#.set('spark.executor.heartbeatInterval','1000000s')
        #sc = SparkContext(conf=conf)
        sc = SparkContext("local", "simdownload")
        sqlContext = SQLContext(sc)
        #print(sc._conf.getAll())
        sv_rdd = sc.parallelize(sparsemat.toarray(), numSlices= 1000)
        #populate the values from rdd to dataframe
        dfspark = sv_rdd.map(lambda x: Row(**f(x))).toDF()

        row_with_index = Row(*["id"] + dfspark.columns)
        def make_row(columns):
            def _make_row(row, uid):
                row_dict = row.asDict()
                return row_with_index(*[uid] + [row_dict.get(c) for c in columns])
            return _make_row
        print('ok')

        f = make_row(dfspark.columns)
        # create a new dataframe with id column (use indexes)
        dfidx = (dfspark.rdd.zipWithIndex().map(lambda x: f(*x)).toDF(StructType([StructField("id", LongType(), False)] + dfspark.schema.fields)))
        #compute cosine sim by rows
        pred = IndexedRowMatrix(dfidx.rdd.map(lambda row: IndexedRow(row.id, row[1:])))
        pred1 = pred.toBlockMatrix().transpose().toIndexedRowMatrix()
        pred_sims = pred1.columnSimilarities()
        #convert coordinatematrix (pred_sims) into a dataframe
        columns = ['from', 'to', 'sim']
        vals = pred_sims.entries.map(lambda e: (e.i, e.j, e.value))
        dfsim = sqlContext.createDataFrame(vals, columns)

        print('Sim Done!')
        print('Time Sim Done: ' + time.strftime("%H:%M:%S"))

        json_data = {}
        for i in range(m):
            target_id = int(dataset_u[i])
            dftemp = dfsim.where((psf.col("from") == target_id) | (psf.col("to") == target_id)).sort(psf.desc("sim")).limit(self.num_top_dataset)
            df = dftemp.toPandas()
            #v = df.iloc[:, :-1].values
            #ii = np.arange(len(df))[:, None]
            #ji = np.argsort(v == target_id, axis=1)  # replace `1` with your ID
            #related_ids = (v[ii, ji][:, 0]).tolist()
            #related_datasets = [dataset_u[i] for i in related_ids]
            myarr = []
            for index, row in df.iterrows():
                from_id = row['from']
                to_id = row['to']
                if (from_id != target_id):
                    myarr.append(int(from_id))
                if (to_id != target_id):
                    myarr.append(int(to_id))
            related_datasets = [dataset_u[i] for i in myarr]
            downloads = download_count.loc[target_id]['count']
            data = {}
            data['related_datasets'] = related_datasets
            data['total_downloads'] = int(downloads)
            json_data[target_id] = data

        print('Time Sim 1: ' + time.strftime("%H:%M:%S"))
        with open("results/downloads.json", 'w') as fp:
            json.dump(json_data, fp)

        print('Time Sim 2: ' + time.strftime("%H:%M:%S"))

        sc.stop()

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
