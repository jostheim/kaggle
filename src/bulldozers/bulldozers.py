import pandas as pd
import numpy as np
import types
import dateutil
import datetime
from dateutil import tz
from multiprocessing import Pool
import os, sys
import random
import pickle
import pytz
from pytz import timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import traceback

def parse_date_time(val):
    if str(val).lower().strip() not in na_values and str(val).lower().strip() != "nan":
        #'2012-11-12 17:30:00+00:00
        try:
            datetime_obj = dateutil.parser.parse(val)
            datetime_obj = datetime_obj.replace(tzinfo=timezone('UTC'))
            datetime_obj = datetime_obj.astimezone(timezone('UTC'))
            return datetime_obj
        except ValueError as e:
#            print e
            return np.nan
    else:
        return np.nan
#        return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")

def flatten(df, column_to_flatten, column_to_sort_flattening=None, max_number_to_flatten=None, prefix=""):
    grouped = df.groupby(column_to_flatten)
    groups = []
    i = 0
    for name, group in grouped:
        d = {column_to_flatten:name}
        if column_to_sort_flattening is not None:
            group = group.sort_index(by=column_to_sort_flattening, ascending=False)
        if max_number_to_flatten is None:
            max_number_to_flatten = len(group.values)
        for k, (ix,row) in enumerate(group.iterrows()[0:max_number_to_flatten]):
            prefix1 = k
            if k == 0:
                prefix1 = "first"
            if k == len(group.values)-1:
                prefix1 = "last"
            d["{0}{1}_{2}".format(prefix, prefix1, "index")] = ix
            for j, val in enumerate(row):
                if group.columns[j] != column_to_flatten:
                    d["{0}{1}_{2}".format(prefix, prefix1, group.columns[j])] = val
        groups.append(d)
        i += 1
    tmp_df = pd.DataFrame(groups)
    tmp_df.set_index(column_to_flatten, inplace=True, verify_integrity=True)
    return tmp_df

if __name__ == '__main__':
    store = pd.HDFStore('bulldozers.h5')
    store_filename = 'bulldozers.h5'
    data_prefix = '/Users/jostheim/workspace/kaggle/data/bulldozers/'
    df = pd.read_csv("{0}{1}".format(data_prefix, "train.csv"), index_col=0, parse_dates=[9], date_parser=parse_date_time)
    unique_sales_dates = np.unique(df['saledate'])
    unique_locations = np.unique(df['state'])
    i = 0
    for sale_date in unique_sales_dates:
        per_sale_df = df[df['saledate'] == sale_date]
        if len(np.unique(per_sale_df['state'])) > 1:
            print "uhoh", len(np.unique(per_sale_df['state']))
        flattened_df = None
        for ix, row in per_sale_df.iterrows():
            t_df = flatten(per_sale_df, 'saledate', 'YearMade', index_to_ignore=ix)
            t_df['SalesID'] = ix
            t_df.set_index('SalesID', inplace=True, verify_integrity=True)
            if flattened_df is None:
                flattened_df = t_df
            else:
                flattened_df = flattened_df.append(t_df)
            print flattened_df
        df = df.join(flattened_df)
        i += 1
        if i > 2:
            break
    print df
        
        
    
    
    