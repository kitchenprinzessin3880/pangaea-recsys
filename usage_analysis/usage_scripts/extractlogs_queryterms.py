import configparser as ConfigParser
import os
import argparse
import re
import pandas as pd
from urllib import parse
from itertools import tee
import time
from datetime import datetime
import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import pickle

class ProcessLogs:
    global dfs;
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--config", required=True,help="Path to usage.ini config file")
        args = ap.parse_args()
        config = ConfigParser.ConfigParser()
        config.read(args.config)
        self.source_folder = config['DATASOURCE']['source_folder']
        self.source_file_prefix = config['DATASOURCE']['source_file_prefix']
        self.source_file_suffix = config['DATASOURCE']['source_file_suffix']
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath('__file__'))))
        self.source_dir = os.path.abspath(os.path.join(self.parent_dir, self.source_folder))
        # read file with data ids
        self.published_datasets = []
        idfile_dir = self.parent_dir + '\\results\\ids.p'
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
        # get request uri, extract data id
        dfmain['_id'] = dfmain['request'].str.extract(r'PANGAEA.\s*([^\n? ]+)', expand=False)
        # remove rows if dataset is NaN
        dfmain = dfmain.dropna(subset=['_id'], how='all')
        # convert time
        dfmain['time'] = dfmain['time'].str.strip('[]').str[:-6]
        #dfmain['time'] = pd.to_datetime(dfmain['time'], format='%d/%b/%Y:%H:%M:%S')
        #dfmain['time_normalize'] = dfmain['time'].dt.date
        dfmain['time'] = pd.to_datetime(dfmain['time'], format='%d/%b/%Y:%H:%M:%S')
        dfmain['time'] = dfmain['time'].dt.date
        return dfmain

    def get_query(self,url):
        qparams = dict(parse.parse_qsl(parse.urlsplit(url).query))
        query_string = ""
        if len(qparams) > 0:
            for key in qparams:
                if re.match(r'f[.]|q|t|p', key):
                    query_string += qparams[key] + " "
        return query_string

    def pairwise(self,iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def parse_datetime(self,x):
        dtime = datetime.strptime(x[1:-7], '%d/%b/%Y:%H:%M:%S')
        d = dtime.date()
        return d

    def readLogs(self):
        dfs = []
        for file in os.listdir(self.source_dir):
            if file.startswith(self.source_file_prefix) and file.endswith(self.source_file_suffix):
                filepath = os.path.join(self.source_dir, file)
                #print(filepath)
                data = pd.read_csv(filepath, compression='bz2', encoding='ISO-8859-1',
                                   sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', engine='python', header=0,
                                   usecols=[0, 3, 4, 5, 7, 8],
                                   names=['ip', 'time', 'request', 'status', 'referer', 'user_agent'],
                                   converters={"request": self.parse_str})

                df = self.cleanLogs(data)
                dfs.append(df)

        # Concatenate all data into one DataFrame
        df_final = pd.concat(dfs, ignore_index=True)
        print("before", str(df_final.shape))
        # exlude rows that contains old data
        df_final = df_final[df_final['_id'].isin(self.published_datasets)]
        print("after", str(df_final.shape))
        return df_final

    def getQueryTerms(self, df_final):
        # identify first and second degree queries
        df_final['query_1'] = df_final['referer'].map(self.get_query)
        df_final['query_2'] = ""
        df_final = df_final[['ip', '_id', 'query_1', 'query_2', 'time']]

        first = df_final.groupby(by=['ip', 'time'])
        first_filtered = first.filter(lambda x: len(x[x['query_1'] != ""]) > 0)
        second = first_filtered.groupby(by=['ip', 'time'])
        filtered = second.filter(lambda x: len(x[x['query_1'] == ""]) > 0)

        for (i1, row1), (i2, row2) in self.pairwise(filtered.iterrows()):
            if ((row1["query_1"] != "") and (row2["query_1"] == "")):
                filtered.set_value(i2, 'query_2', row1["query_1"])

        filtered = filtered[~((filtered.query_1 == "") & (filtered.query_2 == ""))]
        dfgroup = filtered.groupby('_id')['query_1', 'query_2'].apply(lambda x: x.sum())

        # strip white spaces
        dfgroup['query_1'] = dfgroup['query_1'].str.strip()
        dfgroup['query_2'] = dfgroup['query_2'].str.strip()
        dfgroup.loc[dfgroup.query_2 == "", 'query_2'] = None
        dfgroup.loc[dfgroup.query_1 == "", 'query_1'] = None
        return dfgroup

def main():
    start_time = time.time()
    print ('Start Time: '+time.strftime("%H:%M:%S"))
    c1 = ProcessLogs()
    df_final = c1.readLogs()
    # only select referer related to pangaea, get query terms for each datasets
    domains = ['doi.pangaea.de', 'www.pangaea.de', '/search?']
    domains_joins = '|'.join(map(re.escape, domains))
    df_final = df_final[(df_final.referer.str.contains(domains_joins))]
    query_df = c1.getQueryTerms(df_final)
    query_df = query_df.reset_index()
    query_df = query_df.set_index('_id')

    print('Total datasets:'+str(query_df.shape))
    print(str(query_df.info()))

    result_dir = c1.parent_dir + '\\results'
    query_df.to_json(result_dir+'\\queries.json',orient='index')
    secs =  (time.time() - start_time)
    print('Total Execution Time: '+str(dt.timedelta(seconds=secs)))

if __name__== "__main__":
  main()

