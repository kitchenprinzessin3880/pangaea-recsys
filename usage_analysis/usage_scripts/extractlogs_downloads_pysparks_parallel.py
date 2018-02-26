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
from pyspark.sql.types import Row
import pyspark.sql.functions as psf
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark import SparkConf,SparkContext,SQLContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import logging
import multiprocessing
from pandas.api.types import CategoricalDtype

class ProcessLogs:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--config", required=True,help="Path to usage.ini config file")
        args = ap.parse_args()
        config = ConfigParser.ConfigParser()
        config.read(args.config)
        self.parent_dir = config['GLOBAL']['main_dir']
        self.source_file_prefix = config['DATASOURCE']['source_file_prefix']
        self.source_file_suffix = config['DATASOURCE']['source_file_suffix']
        self.num_top_dataset = int(config['DATASOURCE']['number_of_reldatasets'])
        self.source_dir = os.path.join(self.parent_dir, config['DATASOURCE']['source_folder'])
        self.DATA_LIST_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['datalist_file'])
        self.HDF_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['hdf_file'])
        self.JSONUSAGE_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['usage_file'])
        self.PUBLISHED_DATA_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['published_data_file'])
        self.LOG_FILE = os.path.join(self.parent_dir, config['DATASOURCE']['log_file'])

        logging.basicConfig(level=logging.INFO, filename=self.LOG_FILE, filemode="a+",
                            format="%(asctime)-15s %(levelname)-8s %(message)s")
        logging.info('Start Time: ' + time.strftime("%H:%M:%S"))

        # read file with data ids
        self.published_datasets = []
        # idfile_dir = self.parent_dir + '/usage_scripts/results/ids.p'
        with open(self.PUBLISHED_DATA_FILE, 'rb') as fp:
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

    # wrap your csv importer in a function that can be mapped
    def read_csv(self, filename):
        data = pd.read_csv(filename, compression='bz2', encoding='ISO-8859-1',
                           sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', engine='python', header=0,
                           usecols=[0, 3, 4, 5, 8],
                           names=['ip', 'time', 'request', 'status', 'user_agent'],
                           converters={"request": self.parse_str})
        df = self.cleanLogs(data)
        return df

    def readLogs(self):
        dfs = []
        dirs = os.path.join(self.parent_dir, self.source_dir)
        # set up your pool
        pool = multiprocessing.Pool()  # or whatever your hardware can support
        logging.info("CPU count: %s ", str(pool._processes))

        # get a list of file names
        files = os.listdir(dirs)
        file_list = [os.path.join(self.source_dir, filename) for filename in files if
                     filename.startswith(self.source_file_prefix) and filename.endswith(self.source_file_suffix)]

        # have your pool map the file names to dataframes
        df_list = pool.map(self.read_csv, file_list)

        # Concatenate all data into one DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)
        logging.info("Before:%s ", str(combined_df.shape))
        # exlude rows that contains old data
        df_final = combined_df[combined_df['_id'].isin(self.published_datasets)]
        logging.info("After - remove old datasets %s:", str(df_final.shape))
        pool.close()
        logging.info('Finish Combining Logs')
        return df_final

    def get_Total_Related_Downloads(self,dfmain):
        #total downloads
        download_count = dfmain.groupby(['_id'])['_id'].agg(['count'])

        #build datasets vs ip similarity matrix
        group = pd.DataFrame({'download_count': dfmain.groupby(['_id', 'ip']).size()}).reset_index()
        person_u = list(group.ip.unique())
        dataset_u = list(group._id.unique())

        outF = open(self.DATA_LIST_FILE, "w")
        for line in dataset_u:
            outF.write(str(line))
            outF.write("\n")
        outF.close()

        data = group['download_count'].tolist()
        row = group._id.astype('category', categories=dataset_u).cat.codes
        cols = group.ip.astype('category', categories=person_u).cat.codes

        len_dataset = len(dataset_u)
        len_person =  len(person_u)
        logging.info("Datasets vs Ips :%s %s", str(len_dataset), str(len_person))  # 310177 x 81650
        sparsemat = sparse.csr_matrix((data, (row, cols)), dtype=np.int,shape=(len_dataset, len_person))
        logging.info('Sparse matrix size in bytes:%s', str(sparsemat.data.nbytes))
        m, n = sparsemat.shape

        def f(x):
            d = {}
            for i in range(len(x)):
                d[str(i)] = float(x[i])
            return d

        # load PySpark using findSpark package

        #SparkContext.setSystemProperty('spark.executor.memory', '5g')
        #SparkContext.setSystemProperty('spark.driver.memory', '5g')
        #SparkContext.setSystemProperty('spark.executor.heartbeatInterval', '1000000000s')
        #findspark.init()
        conf = SparkConf().setAppName("sim_download").set('spark.local.dir', 'C://Users//dev087//MyClustering//pangaea-recsys//usage_analysis')
        #conf = (conf.setMaster('local[*]').set('spark.executor.memory', '4G'))#.set('spark.executor.heartbeatInterval','1000000s')
        sc = SparkContext(conf=conf)
        #sc = SparkContext("local", "simdownload")
        #sc = SparkContext(appName= "sim_download")
        #sc.getConf.set('spark.local.‌​dir', 'C://Users//dev087//MyClustering//pangaea-recsys//usage_analysis')
        sqlContext = SQLContext(sc)
        #print(str(sc._conf.getAll()))
        logging.info('Parallelize Sparse Array..')
        sv_rdd = sc.parallelize(sparsemat.toarray())
        #populate the values from rdd to dataframe
        dfspark = sv_rdd.map(lambda x: Row(**f(x))).toDF()
        logging.info('Parallelize Sparse Map Partition Num: %s', str(dfspark.rdd.getNumPartitions()))

        row_with_index = Row(*["id"] + dfspark.columns)
        def make_row(columns):
            def _make_row(row, uid):
                row_dict = row.asDict()
                return row_with_index(*[uid] + [row_dict.get(c) for c in columns])
            return _make_row
        logging.info('Parallelize Sparse Array - Done..')

        f = make_row(dfspark.columns)
        # create a new dataframe with id column (use indexes)
        dfidx = (dfspark.rdd.zipWithIndex().map(lambda x: f(*x)).toDF(StructType([StructField("id", LongType(), False)] + dfspark.schema.fields)))
        #compute cosine sim by rows
        logging.info('create a new dataframe with id column Done!')
        pred = IndexedRowMatrix(dfidx.rdd.map(lambda row: IndexedRow(row.id, row[1:])))
        pred1 = pred.toBlockMatrix().transpose().toIndexedRowMatrix()
        pred_sims = pred1.columnSimilarities()
        logging.info('columnSimilarities Done!')
        #convert coordinatematrix (pred_sims) into a dataframe
        columns = ['from', 'to', 'sim']
        vals = pred_sims.entries.map(lambda e: (e.i, e.j, e.value))
        dfsim = sqlContext.createDataFrame(vals, columns)

        logging.info('Sim Compute Final Done!')

        json_data = {}
        for i in range(m):
            target_id = int(dataset_u[i])
            dftemp = dfsim.where((psf.col("from") == i) | (psf.col("to") == i)).sort(psf.desc("sim")).limit(self.num_top_dataset)
            df = dftemp.toPandas()
            # v = df.iloc[:, :-1].values
            # ii = np.arange(len(df))[:, None]
            # ji = np.argsort(v == i, axis=1)  # replace `1` with your ID
            # related_ids = (v[ii, ji][:, 0]).tolist()
            # related_datasets = [dataset_u[i] for i in related_ids]
            myarr = []
            for index, rw in df.iterrows(): #this is a bit faster than numpy above
                from_id = rw['from']
                to_id = rw['to']
                if (from_id != i):
                    myarr.append(int(from_id))
                if (to_id != i):
                    myarr.append(int(to_id))
            related_datasets = [int(dataset_u[i]) for i in myarr]

            downloads = download_count.loc[target_id]['count']
            data = {}
            data['related_datasets'] = related_datasets
            data['total_downloads'] = int(downloads)
            json_data[target_id] = data

        logging.info('Writing recsys to json...')
        with open(self.JSONUSAGE_FILE, 'w') as fp:
            json.dump(json_data, fp)

        logging.info('Writing recsys to json done!')
        sc.stop()


def main():
    start_time = time.time()
    c1 = ProcessLogs()
    dfdwl = c1.readLogs()
    #get total downloads and sim by download profiles
    dfdwl = dfdwl.drop_duplicates(['time', 'ip', '_id'])
    dfdwl = dfdwl[['ip', '_id']]
    c1.get_Total_Related_Downloads(dfdwl)

    secs =  (time.time() - start_time)
    logging.info('Total Execution Time: '+str(dt.timedelta(seconds=secs)))

if __name__ == "__main__":
    main()

