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
    if val is not np.nan:
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

def flatten(df, column_to_flatten, column_to_sort_flattening=None, index_to_ignore=None, max_number_to_flatten=None, prefix=""):
    print "grouping"
    grouped = df.groupby(column_to_flatten)
    groups = []
    i = 0
    for name, group in grouped:
        d = {column_to_flatten:name}
        if column_to_sort_flattening is not None:
            print "sorting"
            group = group.sort_index(by=column_to_sort_flattening, ascending=False)
        if max_number_to_flatten is None:
            max_number_to_flatten = len(group.values)
        for k, (ix,row) in enumerate(group.iterrows()):
            if index_to_ignore is not None and index_to_ignore == ix:
                continue
            prefix1 = k
            if k == 0:
                prefix1 = "first"
            if k == len(group.values)-1:
                prefix1 = "last"
            for j, val in enumerate(row):
                if group.columns[j] != column_to_flatten:
                    d["{0}{1}_{2}".format(prefix, prefix1, group.columns[j])] = val
            if k > max_number_to_flatten:
                break
        groups.append(d)
        i += 1
    tmp_df = pd.DataFrame(groups)
    tmp_df.set_index(column_to_flatten, inplace=True, verify_integrity=True)
    return tmp_df


def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column],
        "SaleDayOfWeek": [d.weekday for d in date_column]
        }, index=date_column.index)

def flatten_data_at_same_auction(df):
    unique_sales_dates = np.unique(df['saledate'])
    i = 0
    new_df = df.copy(True)
    for sale_date in unique_sales_dates:
        per_sale_df = df[df['saledate'] == sale_date]
        unique_states = np.unique(per_sale_df['state'])
        for state in unique_states:
            per_sale_df = per_sale_df[per_sale_df['state'] == state]
            flattened_df = None
            for ix, row in per_sale_df.iterrows():
                print "flattening"
                t_df = flatten(per_sale_df, 'saledate', 'YearMade', index_to_ignore=ix)
                t_df['SalesID'] = ix
                t_df.set_index('SalesID', inplace=True, verify_integrity=True)
                if flattened_df is None:
                    flattened_df = t_df
                else:
                    flattened_df = flattened_df.append(t_df)
            if flattened_df is not None:
                print "joining flattened", new_df
                new_df = new_df.join(flattened_df)
        i += 1
        print "{0}/{1}".format(i, len(unique_sales_dates))

def convert_categorical_to_features(train, test, columns, train_fea, test_fea):
    for col in columns:
        if train[col].dtype == np.dtype('object'):
            s = np.unique(train[col].values)
            mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
#            print mapping
            train_fea = train_fea.join(train[col].map(mapping))
            test_fea = test_fea.join(test[col].map(mapping))
        else:
            train_fea = train_fea.join(train[col])
            test_fea = test_fea.join(test[col])
    return train_fea, test_fea


if __name__ == '__main__':
    store = pd.HDFStore('bulldozers.h5')
    store_filename = 'bulldozers.h5'
    data_prefix = '/Users/jostheim/workspace/kaggle/data/bulldozers/'
    train = pd.read_csv("{0}{1}".format(data_prefix, "Train.csv"), 
                        converters={"saledate": dateutil.parser.parse})
    test = pd.read_csv("{0}{1}".format(data_prefix, "Valid.csv"),  
                       converters={"saledate": dateutil.parser.parse})
    
    train.fillna("NaN", inplace=True)
    test.fillna("NaN", inplace=True)
    
    train_fea = get_date_dataframe(train["saledate"])
    test_fea = get_date_dataframe(test["saledate"])
    
    columns = set(train.columns)
    columns.remove("SalesID")
    columns.remove("SalePrice")
#    columns.remove("saledate")
    
    train_fea, test_fea = convert_categorical_to_features(train, test, columns, train_fea, test_fea)
    train_fea = flatten_data_at_same_auction(train_fea)
    train_fea.to_csv("train.csv")
    train = None
    test_fea = flatten_data_at_same_auction(test_fea)
    test_fea.to_csv("test.csv")
    test = None
    
        
        
    
    
    