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
import os

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
        #idfile_dir = parent_dir + '\\usage_scripts\\results\\ids.p'
        #with open(idfile_dir, 'rb') as fp:
            #self.published_datasets = pickle.load(fp)

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
        # exlude rows that contains old data
        #df_final = df_final[df_final['_id'].isin(self.published_datasets)]

        return df_final

    def getQueryTerms(self, df_final):
        # identify first and second degree queries
        df_final['query_1'] = df_final['referer'].map(self.get_query)
        df_final = df_final[['_id', 'query_1']]
        isEmptyString =  "" in df_final.query_1.unique()
        if (isEmptyString):
            df_final = df_final[df_final.query_1 != ""]
            print("Shape after Empty", str(df_final.shape))

        dfgroup = df_final.groupby(['query_1'])['_id'].apply(list).reset_index(name='datasets')
        dfgroup['Length'] = dfgroup['datasets'].str.len()
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
    print("Min, Max Time : ", df_final.time.min(), df_final.time.max())
    query_df = c1.getQueryTerms(df_final)

    print('Total datasets:'+str(query_df.shape))
    result_path = os.path.abspath(os.path.join(c1.parent_dir, "results"))
    query_df.to_csv(result_path+'\\query_data_frequency.csv', sep='\t', encoding='utf-8')
    secs =  (time.time() - start_time)
    print('Total Execution Time: '+str(dt.timedelta(seconds=secs)))

if __name__== "__main__":
  main()