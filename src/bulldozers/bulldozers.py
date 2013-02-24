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
import csv



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

def flatten_data_at_same_auction(df):
    i = 0
    new_df = df.copy(True)
    new_df.set_index('SalesID', inplace=True, verify_integrity=True)
    unique_sales_dates = np.unique(df['saledate'])
    for col, series in new_df.iteritems():
        new_df[col] = series.apply(lambda x: replace_nan_with_random(x))
    flattened_df = None
    for sale_date in unique_sales_dates:
        per_sale_df = new_df[new_df['saledate'] == sale_date]
        unique_states = np.unique(per_sale_df['state'])
        for state in unique_states:
            per_sale_per_state_df = per_sale_df[per_sale_df['state'] == state]
            rows = []
            for k, (ix, row) in enumerate(per_sale_per_state_df.iterrows()):
                cosines = cosine_similarity(per_sale_per_state_df, row)
                cosines_t = []
                for cosine in cosines:
                    cosines_t.append(cosine[0])
                series = pd.Series(np.array(cosines_t), index=per_sale_per_state_df.index, name="similarity")
                per_sale_per_state_df_copy = per_sale_per_state_df.copy(True)
                per_sale_per_state_df_copy['similarity'] = series
                per_sale_per_state_df_copy = per_sale_per_state_df_copy.sort_index(by='similarity', ascending=False)
                d = {"SalesID":ix}
                for kk, (ix_1, row_1) in enumerate(per_sale_per_state_df_copy.iterrows()):
                    if ix_1 != ix:
                        for col, val in row_1.iteritems():
                            d["{0}_{1}".format(kk, col)] = val 
                if len(d.keys()) > 1:
                    rows.append(d)
                print "inner: {0}/{1}".format(k, len(per_sale_df))
            if len(rows) > 1:
                tmp_df = pd.DataFrame(rows)
                tmp_df.set_index('SalesID', inplace=True, verify_integrity=True)
                if flattened_df is None:
                    flattened_df = tmp_df
                else:
                    flattened_df.append(tmp_df)
        i += 1
        print "{0}/{1}".format(i, len(unique_sales_dates))
    if flattened_df is not None and len(flattened_df) > 1:
#        flattened_df = pd.DataFrame(flattened_df)
#        flattened_df.set_index('SalesID', inplace=True, verify_integrity=True)
        print "joining flattened", new_df
        new_df = new_df.join(flattened_df)
    return new_df

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
            if col in test.columns:
                test_fea = test_fea.join(test[col])
    return train_fea, test_fea

def get_related_rows(train_fea_tmp, row, index):
    cosines = cosine_similarity(train_fea_tmp, row)
    cosines_t = []
    for cosine in cosines:
        cosines_t.append(cosine[0])
    series = pd.Series(np.array(cosines_t), index=train_fea_tmp.index, name="similarity")
    series = series.order(ascending=False)
    d = {'index':index}
    i = 0
    for i, (ix, sim) in enumerate(series.iteritems()):
        if ix == index:
            continue
        d["{0}_similarity".format(i)] = sim
        for col, val in train_fea_tmp.ix[ix].iteritems():
            d["{0}_{1}".format(i, col)] = val 
        i += 1
        if i > 50:
            break
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

def replace_nan_with_random(x):
    if str(x).lower().strip() == 'nan' or x is np.nan:
        return random.uniform(0, 1.0)
    else:
        return x

def get_all_related_rows_as_features(fea):
    for col, series in fea.iteritems():
        fea[col] = series.apply(lambda x: replace_nan_with_random(x))
    all_df = None
    pool_queue = []
    processes = 8
    pool = Pool(processes=processes)
    update = len(fea)/1000
    for i, (ix, row) in enumerate(fea.iterrows()):
        pool_queue.append([fea, row, ix])
        if i > 0 and i%update == 0:
            res = pool.map(get_related_rows_proxy, pool_queue, len(pool_queue)/processes)
            result = pd.DataFrame(res)
            result.set_index("index", inplace=True, verify_integrity=True)
            if all_df is None:
                all_df = result
            else:
                all_df.append(result)
            pool_queue = []
            print "done processing {0}/{1}".format(i, len(fea)) 
    if len(pool_queue) > 0:
        res = pool.map(get_related_rows_proxy, pool_queue, len(pool_queue)/processes)
        result = pd.DataFrame(result)
        if all_df is None:
            all_df = result
        else:
            all_df.append(result)
    pool.terminate()
    return all_df


def prepare_test_features(data_prefix):
    train = pd.read_csv("{0}{1}".format(data_prefix, "Train.csv"), 
                        converters={"saledate": dateutil.parser.parse})
    test = pd.read_csv("{0}{1}".format(data_prefix, "Valid.csv"),  
                       converters={"saledate": dateutil.parser.parse})
    machine_appendix = pd.read_csv("{0}{1}".format(data_prefix, "Machine_Appendix.csv"), index_col=0)
    test['MfgYear'] = np.nan
    test['fiManufacturerID'] = np.nan
    test['fiManufacturerDesc'] = np.nan
    test['PrimarySizeBasis'] = np.nan
    test['PrimaryLower'] = np.nan
    test['PrimaryUpper'] = np.nan
    for ix, row in test.iterrows():
        machine_id = row['MachineID']
        machine_appendix_row = machine_appendix.ix[machine_id]
        for col, val in machine_appendix_row.iteritems():
            row[col] = val 
    test['YearMade'] = test['YearMade'].apply(lambda x: x if x != 1000 else np.nan)
    test.fillna("NaN", inplace=True)
    test_fea = get_date_dataframe(test["saledate"])
    train_fea = get_date_dataframe(train["saledate"])
    columns = set(train.columns)
#    columns.remove("SalesID")
#    columns.remove("SalePrice")
    columns.remove("saledate")
    train_fea, test_fea = convert_categorical_to_features(train, test, columns, train_fea, test_fea)
    # gotta put nan's back
    for col, series in test_fea.iteritems():
        test_fea[col] = series.apply(lambda x: x if x != "NaN" else np.nan)
    joiner = get_all_related_rows_as_features(test_fea.copy(True))
    test_fea = test_fea.join(joiner)
    return test_fea

def prepare_train_features(data_prefix, sample_size = None, output_bayesian=False):
    train = pd.read_csv("{0}{1}".format(data_prefix, "Train.csv"), 
                        converters={"saledate": dateutil.parser.parse})
    if sample_size is not None:
        train = sample(train, sample_size)
    test = pd.read_csv("{0}{1}".format(data_prefix, "Valid.csv"),  
                       converters={"saledate": dateutil.parser.parse})
    machine_appendix = pd.read_csv("{0}{1}".format(data_prefix, "Machine_Appendix.csv"), index_col=0)
    train['MfgYear'] = np.nan
    train['fiManufacturerID'] = np.nan
    train['fiManufacturerDesc'] = np.nan
    train['PrimarySizeBasis'] = np.nan
    train['PrimaryLower'] = np.nan
    train['PrimaryUpper'] = np.nan
    # clean up data in original with the appendix
    for ix, row in train.iterrows():
        machine_id = row['MachineID']
        machine_appendix_row = machine_appendix.ix[machine_id]
        for col, val in machine_appendix_row.iteritems():
            row[col] = val 
    
    test['MfgYear'] = np.nan
    test['fiManufacturerID'] = np.nan
    test['fiManufacturerDesc'] = np.nan
    test['PrimarySizeBasis'] = np.nan
    test['PrimaryLower'] = np.nan
    test['PrimaryUpper'] = np.nan
    for ix, row in test.iterrows():
        machine_id = row['MachineID']
        machine_appendix_row = machine_appendix.ix[machine_id]
        for col, val in machine_appendix_row.iteritems():
            row[col] = val         
    
    train['YearMade'] = train['YearMade'].apply(lambda x: x if x != 1000 else np.nan)
    train.fillna("NaN", inplace=True)
    train_fea = get_date_dataframe(train["saledate"])
    test_fea = get_date_dataframe(test["saledate"])
    
    if output_bayesian:
        bayes_df = train.join(train_fea)
        bayes_df.set_index("SalesID", inplace=True, verify_integrity=True)
        del bayes_df["saledate"]
        header = []
        for col in bayes_df.columns:
            header.append("{0}:{1}".format(col, "string"))
        data = []
        for ix, row in bayes_df.iterrows():
            data.append(row)
        csv_writer = csv.writer(open('train_bayesian.csv', 'w'))
        csv_writer.writerow(header)
        csv_writer.writerows(data)
        csv_writer.close()
        return
    columns = set(train.columns)
#    columns.remove("SalesID")
#    columns.remove("SalePrice") 
#    columns.remove("saledate")
    train_fea, test_fea = convert_categorical_to_features(train, test, columns, train_fea, test_fea)
#    train_fea.set_index("SalesID", inplace=True, verify_integrity=True)
    # gotta put nan's back
    for col, series in train_fea.iteritems():
        train_fea[col] = series.apply(lambda x: replace_string_nan(x))
    train_fea = flatten_data_at_same_auction(train_fea)
#    joiner = get_all_related_rows_as_features(train_fea.copy(True))
#    train_fea = train_fea.join(joiner)
    train_fea.to_csv("train.csv")
    return train_fea

def replace_string_nan(x):
    if str(x).lower().strip() == "nan" or x is np.nan:
        return np.nan
    else:
        return x

def get_metric(cfr, features, targets):
    #Mean Square Log Error MSLE = (1/N) * SUM(log(y)est - log(y)actual)2
    #Root Mean Square Log Error RMSLE = (MSLE)1/2
    sum_diff = 0.0
    p = cfr.predict_proba(features)
    unique_classes = sorted(cfr.classes_[0])
    for k, target in enumerate(targets):
        # expectation across all classes
        expectation = np.sum(unique_classes*p[k])
        print expectation, target
        sum_diff += (np.power((np.log(expectation) - np.log(target)),2))
    mean_diff = np.sqrt(sum_diff/float(len(targets)))
    return mean_diff


def random_forest_cross_validate(targets, features, nprocesses=-1):
    cv = cross_validation.KFold(len(features), k=5, indices=False)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for i, (traincv, testcv) in enumerate(cv):
        cfr = RandomForestClassifier(
            n_estimators=100,
            max_features=None,
            verbose=2,
            compute_importances=True,
            n_jobs=nprocesses,
            random_state=0,
        )
        print "Fitting cross validation #{0}".format(i)
        cfr.fit(features[traincv], targets[traincv])
        print "Scoring cross validation #{0}".format(i)
        cfr.set_params(n_jobs=1) # read in the features to predict, remove bad columns
        score = cfr.score(features[testcv], targets[testcv])
        print "Score for cross validation #{0}, score: {1}".format(i, score)
        mean_diff = get_metric(cfr, features[testcv], targets[testcv])
        print "Mean difference: {0}".format(mean_diff)
        results.append(mean_diff)
        print "Features importance"
        features_list = []
        for j, importance in enumerate(cfr.feature_importances_):
            if importance > 0.0:
                column = features.columns[j]
                features_list.append((column, importance))
        features_list = sorted(features_list, key=lambda x: x[1], reverse=True)
        for j, tup in enumerate(features_list):
            print j, tup
        pickle.dump(features_list, open("important_features.p", 'wb'))
        print "Mean difference: {0}".format(mean_diff)
        results.append(mean_diff)

def sample(df, nsamples):
    rows = random.sample(df.index, sample_size)
    df = df.ix[rows]
    return df

if __name__ == '__main__':
    data_prefix = '/Users/jostheim/workspace/kaggle/data/bulldozers/'
    kind = sys.argv[1]
    sample_size = None
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    nprocesses = 8
    if len(sys.argv) > 3:
        nprocesses = int(sys.argv[3])
    if kind == "prepare_test_features":
        prepare_test_features(data_prefix)
    if kind == "prepare_train_features":
        prepare_train_features(data_prefix)
    if kind == "write_bayesian":
        prepare_train_features(data_prefix, sample_size, True)
    if kind == "fix_train_features":
        train_df = pd.read_csv("train.csv", index_col=0)
        train = pd.read_csv("{0}{1}".format(data_prefix, "Train.csv"), 
                        converters={"saledate": dateutil.parser.parse})
        train_df = train_df.join(train['SalePrice'])
        train_df.to_csv("train.csv")
    if kind == "cross_validate":
        train_df = pd.read_csv("train.csv", index_col=0)
        if sample_size is not None:
            train_df = sample(train_df, sample_size)
        targets = train_df['SalePrice'].dropna()
        features = train_df.ix[targets.index]
        del features['SalePrice']
        print "Doing cross validation with {0} features and {1} targets ".format(len(features), len(targets))
        random_forest_cross_validate(targets, features, nprocesses)
    
        
    
    
    