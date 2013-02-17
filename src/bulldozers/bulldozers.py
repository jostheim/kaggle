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
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot

random.seed(0)


def cosine_similarity(X, Y=None):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Parameters
    ----------
    X : array_like, sparse matrix
        with shape (n_samples_X, n_features).

    Y : array_like, sparse matrix (optional)
        with shape (n_samples_Y, n_features).

    Returns
    -------
    kernel matrix : array_like
        An array with shape (n_samples_X, n_samples_Y).
    """
    # to avoid recursive import

    X, Y = check_pairwise_arrays(X, Y)

    X_normalized = normalize(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize(Y, copy=True)

    K = linear_kernel(X_normalized, Y_normalized)

    return K


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

def flatten(grouped, column_to_flatten, column_to_sort_flattening=None, index_to_ignore=None, max_number_to_flatten=None, prefix=""):
    groups = []
    i = 0
    for name, group in grouped:
        if len(group) == 0:
            continue
        d = {}
#        d = {column_to_flatten:name}
        if column_to_sort_flattening is not None:
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
    return groups


def get_date_dataframe(date_column):
    return pd.DataFrame({
        "SaleYear": [d.year for d in date_column],
        "SaleMonth": [d.month for d in date_column],
        "SaleDay": [d.day for d in date_column],
        "SaleDayOfWeek": [d.weekday() for d in date_column]
        }, index=date_column.index)

def flatten_data_at_same_auction(df):
    unique_sales_dates = np.unique(df['saledate'])
    i = 0
    new_df = df.copy(True)
    new_df.set_index('SalesID', inplace=True, verify_integrity=True)
    flattened_df = []
    for sale_date in unique_sales_dates:
        per_sale_df = df[df['saledate'] == sale_date]
        unique_states = np.unique(per_sale_df['state'])
        for state in unique_states:
            per_sale_df = per_sale_df[per_sale_df['state'] == state]
            
            grouped = per_sale_df.groupby('saledate')
            for k, (ix, row) in enumerate(per_sale_df.iterrows()):
                groups = flatten(grouped, 'saledate', 'YearMade', index_to_ignore=ix)
                for group in groups:
                    group['SalesID'] = ix
#                t_df.set_index('SalesID', inplace=True, verify_integrity=True)
                if flattened_df is None:
                    flattened_df = groups
                else:
                    flattened_df += groups
                print "inner: {0}/{1}".format(k, len(per_sale_df))
        i += 1
        print "{0}/{1}, len(flattened):{2}".format(i, len(unique_sales_dates), len(flattened_df))
    if flattened_df is not None and len(flattened_df) > 1:
        flattened_df = pd.DataFrame(flattened_df)
        flattened_df.set_index('SalesID', inplace=True, verify_integrity=True)
        print "joining flattened", new_df
        new_df = new_df.join(flattened_df)

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

def get_related_rows(train_fea_tmp, row, index):
    cosines = cosine_similarity(train_fea_tmp, row)
    cosines_t = []
    for cosine in cosines:
        cosines_t.append(cosine[0])
    series = pd.Series(np.array(cosines_t), index=train_fea_tmp.index)
    series = series.sort_index(ascending=False)
    d = {'index':index}
    for i, ix in enumerate(series.index[0:100]):
        for col, series in train_fea_tmp.iteritems():
            d["{0}_{1}".format(i, col)] = series.ix[ix] 
    return d

def get_related_rows_proxy(args):
    train_fea_tmp = args[0]
    row = args[1]
    ix = args[2]
    ret = None
    try:
        ret = get_related_rows(train_fea_tmp, row, ix)
    except Exception as e:
        print e
        print traceback.format_exc()
    return ret

def get_all_related_rows_as_features(fea):
    fea_tmp = fea.fillna(random.uniform(0, 4*len(fea)))
    join_dicts = []
    pool_queue = []
    pool = Pool(processes=8)
    results = []
    update = len(fea_tmp)/1000
    for i, (ix, row) in enumerate(fea_tmp.iterrows()):
        pool_queue.append([fea_tmp, row, ix])
        if i%update == 0:
            results += pool.map(get_related_rows_proxy, pool_queue, len(fea_tmp)/8)
            pool_queue = []
            print "done processing {0}/{1}".format(i, len(fea_tmp))
    if len(pool_queue) > 0:
        results += pool.map(get_related_rows_proxy, pool_queue, len(fea_tmp)/8)
    for d in results:
        join_dicts.append(d)
    join_df = pd.DataFrame(join_dicts)
    join_df.set_index('index', inplace=True, verify_integrity=True)
    fea = fea.join(join_df)
    return fea


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
#    columns.remove("SalesID")
    columns.remove("SalePrice")
    columns.remove("saledate")
    
    train_fea, test_fea = convert_categorical_to_features(train, test, columns, train_fea, test_fea)
    
    train_fea = get_all_related_rows_as_features(train_fea.copy(True))
    
    train_fea.to_csv("train.csv")
    train = None

    test_fea = get_all_related_rows_as_features(test_fea.copy(True))
    
    test_fea = flatten_data_at_same_auction(test_fea)
    test_fea.to_csv("test.csv")
    test = None
    
        
        
    
    
    