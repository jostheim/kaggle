'''
Created on Mar 18, 2013

@author: jostheim
'''

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
import matplotlib.pyplot as plt
import csv

name_map = {
            "Colorado State":"Colorado St.",
            "Oklahoma State":"Oklahoma St.",
            "New Mexico State":"New Mexico St.",
            "Middle Tennessee St.":"Middle Tennessee",
            "St. Mary's (Cal.)":"St. Mary's",
            "Michigan State":"Michigan St.",
            "Albany (N.Y.)":"Albany",
            "UNLV":"Nevada Las Vegas",
            "Miami (Fla.)":"Miami FL",
            "Loyola (Md.)":"Loyola MD",
            "Texas-San Antonio":"Texas San Antonio",
            "UNC Asheville":"NC Asheville",
            "Arkansas-Little Rock":"Arkansas Little Rock",
            "Alabama-Birmingham":"Alabama Birmingham",
            "Arkansas-Pine Bluff":"Arkansas Pine Bluff",
            "UTEP":"Texas El Paso",
            "LSU":"Louisiana St.",
            "Texas-Arlington":"Texas Arlington",
            "Maryland-Baltimore County":"MD Baltimore County",
            "Miami (Ohio)":"Miaimi OH",
            "Texas A&M-Corpus Christi":"Texas A&M Corpus Chris",
            "Penn":"Pennsylvania",
            "Central Connecticut St.":"Central Connecticut",
            "Wisconsin-Milwaukee":"Wisconsin Milwaukee",
            "UNC Wilmington":"NC Wilmington",
            "Louisiana-Lafayette":"Louisiana Lafayette",
            "Illinois-Chicago":"Illinois Chicago",
            }


def smart_split_teams(x):
    splits = x.split(" ")
    if splits[len(splits)-1].isdigit():
        t = " ".join(x.split(" ")[0:len(x.split(" "))-1]).strip()
        return t
    return x

def check_names(all_results, all_kpom):
    for ix, row in all_results.iterrows():
        tmp = len(all_kpom[all_kpom['team'] == row['TEAM1']])
        if tmp == 0:
            if row['TEAM1'] not in name_map.keys():
                print "Team1: {0} name not in original".format(row['TEAM1'])
        tmp = len(all_kpom[all_kpom['team'] == row['TEAM2']])
        if tmp == 0:
            if row['TEAM2'] not in name_map.keys():
                print "Team2: {0} name not in original".format(row['TEAM2'])

def generate_final_data(data_prefix, all_kpom, all_results):
    all_kpom['team'] = all_kpom['team'].apply(lambda x:smart_split_teams(x))
    # don't have conf_rate for current year
    if 'conf_rate' in all_kpom.columns:
        del all_kpom['conf_rate']
    # we can get bid if we need to
    if 'bid' in all_kpom.columns:
        del all_kpom['bid']
    all_results['TEAM1'] = all_results['TEAM1'].apply(lambda x:x if "State" not in x else x.replace("State", "St."))
    all_results['TEAM2'] = all_results['TEAM2'].apply(lambda x:x if "State" not in x else x.replace("State", "St."))
    all_results['TEAM1'] = all_results['TEAM1'].apply(lambda x:x if x not in name_map.keys() else name_map[x])
    all_results['TEAM2'] = all_results['TEAM2'].apply(lambda x:x if x not in name_map.keys() else name_map[x])
    join_data = []
    for ix, row in all_results.iterrows():
        final_row = {"index":ix, "ROUND":row["ROUND"], "SEED1":row["SEED1"], "SEED2":row["SEED2"], "SCORE1":row["SCORE1"], "SCORE2":row["SCORE2"], "TEAM1":row["TEAM1"], "TEAM2":row["TEAM2"]}
#        if row["SCORE1"] != "TBD":
#            final_row["WINNER"] = float(row["SCORE1"] - row["SCORE2"])
#        else:
#            final_row["WINNER"] = "-1000"
        if row["SCORE1"] > row["SCORE2"]:
            final_row["WINNER"] = 1
        else:
            final_row["WINNER"] = 2
        tmp = all_kpom[all_kpom['team'] == row['TEAM1']]
        tmp = tmp[tmp['year'] == row['YEAR']]
        if len(tmp) == 0:
            print row['TEAM1']
            continue
        for col, val in tmp.iteritems():
            final_row["{0}_{1}".format("TEAM1", col)] = val.values[0]
        tmp = all_kpom[all_kpom['team'] == row['TEAM2']]
        tmp = tmp[tmp['year'] == row['YEAR']]
        if len(tmp) == 0:
            print row['TEAM2']
            continue
        for col, val in tmp.iteritems():
            final_row["{0}_{1}".format("TEAM2", col)] = val.values[0]
        join_data.append(final_row)
    final_df = pd.DataFrame(join_data)
    final_df.set_index("index", inplace=True, verify_integrity=True)
    final_df.convert_objects()
    final_df.to_csv("joined.csv")
    return final_df

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

def random_forest_learn(targets, features):
    cfr = RandomForestClassifier(
        n_estimators=1000,
        max_features=None,
        verbose=0,
        compute_importances=True,
        n_jobs=6,
        random_state=0,
        )
    cfr.fit(features, targets)
    pickle.dump(cfr, open("model.p", 'wb'))
    return cfr

def random_forest_cross_validate(targets, features, nprocesses=-1):
    num_cv = 5
    cv = cross_validation.KFold(len(features), k=num_cv, indices=False)
    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    score_sum = 0.0
    testcvs = None
    for i, (traincv, testcv) in enumerate(cv):
        cfr = RandomForestClassifier(
            n_estimators=100,
            max_features=None,
            verbose=0,
            compute_importances=True,
            n_jobs=nprocesses,
            random_state=0,
        )
        print "Fitting cross validation #{0}".format(i)
        cfr.fit(features[traincv], targets[traincv])
        print "Scoring cross validation #{0}".format(i)
        cfr.set_params(n_jobs=1)
        predicted = cfr.predict(features[testcv])
        p = cfr.predict_proba(features[testcv])
        score = cfr.score(features[testcv], targets[testcv])
        score_sum += score
        # add stuff to the dataframe so we can plot things
        summer = 0.0
        for j, pred in enumerate(predicted):
            print predicted[j], targets[testcv][j]
            summer += np.power((predicted[j] - targets[testcv][j]), 2)
        print "score error: {0}".format(np.sqrt(summer)/len(testcv))
        testcv = pd.DataFrame(features[testcv])
        testcv['prediction'] = np.nan
        testcv['prob'] = np.nan
        for j, (ix, row) in enumerate(testcv.iterrows()):
            print predicted[j], targets[testcv][j]
            testcv['prediction'].ix[ix] = predicted[j]
            if predicted[j] == 1:
                testcv['prob'].ix[ix] = p[j][0]
            else:
                testcv['prob'].ix[ix] = p[j][1]
        if testcvs is None:
            testcvs = testcv
        else:
            testcvs.append(testcv)
        print "Score for cross validation #{0}, score: {1}".format(i, score)
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
    print "Average Accuracy: {0}".format(float(score_sum)/float(num_cv))
    return testcvs

def get_predictions(cfr, features):
    max_prob = []
    predicted = cfr.predict(features)
    p = cfr.predict_proba(features)
    unique_classes = sorted(cfr.classes_[0])
    expectations = []
    probs = []
    for k, (ix, row) in enumerate(features.iterrows()):
        expectation = np.sum(unique_classes*p[k])
        probs.append(p[k])
        expectations.append(expectation)
        max_prob.append(predicted[k])
    return expectations, max_prob, probs

def get_metric(cfr, features, targets):
    sum_diff = 0.0
    p = cfr.predict_proba(features)
    unique_classes = sorted(cfr.classes_[0])
    for k, target in enumerate(targets):
        # expectation across all classes
        expectation = np.sum(unique_classes*p[k])
        print expectation, target
        sum_diff += np.sqrt(np.power((expectation - target),2))
    mean_diff = sum_diff/float(len(targets))
    return mean_diff

def do_cross_validate(data_prefix):
    kpom_2012 = pd.read_csv("{0}{1}".format(data_prefix, "2012_ncaa_kpom.csv"), index_col=1)
    rpi_2012 = pd.read_csv("{0}{1}".format(data_prefix, "2012_rpi.csv"), index_col=1)
    for ix in kpom_2012.index:
        if ix not in rpi_2012.index:
            print ix
    kpom_2012 = kpom_2012.join(rpi_2012)
    kpom_2012['w_per'] = kpom_2012['w'] / (kpom_2012['w'] + kpom_2012['l'] + 0.0)
    kpom_2012['year'] = 2012
    kpom_2012['tour'] = 'ncaa'
    kpom_2012 = kpom_2012.reset_index()
    kpom_all = pd.read_csv("{0}{1}".format(data_prefix, "ncaa_all_kenpom.csv"))
    kpom_all = kpom_all.append(kpom_2012)
    all_results = pd.read_csv("{0}{1}".format(data_prefix, "ncaa_2013_results 1st_round_play_in.csv"))
    final_df = generate_final_data(data_prefix, kpom_all, all_results)
    final_df.to_csv("final_raw.csv")
    train_fea = pd.DataFrame(index=final_df.index)
    test_fea = pd.DataFrame(index=final_df.index)
    columns = set(final_df.columns)
    train_fea, test_fea = convert_categorical_to_features(final_df, final_df, columns, train_fea, test_fea)
    train_fea.to_csv("final_features.csv")
    del train_fea["SCORE1"]
    del train_fea["SCORE2"]
    targets = train_fea['WINNER']
    del train_fea['WINNER']
    predicted_df = random_forest_cross_validate(targets, train_fea, 1)
    return predicted_df

if __name__ == '__main__':
    data_prefix = '/Users/jostheim/workspace/kaggle/data/ncaa_tourney/'
    if sys.argv[1] == "cross_validate":
        
        predicted_df = do_cross_validate(data_prefix)
        predicted_df.to_csv("predictions.csv")
    elif sys.argv[1] == "fit":
        kpom_2012 = pd.read_csv("{0}{1}".format(data_prefix, "2012_ncaa_kpom.csv"), index_col=1)
        rpi_2012 = pd.read_csv("{0}{1}".format(data_prefix, "2012_rpi.csv"), index_col=1)
        for ix in kpom_2012.index:
            if ix not in rpi_2012.index:
                print ix
        kpom_2012 = kpom_2012.join(rpi_2012)
        kpom_2012['w_per'] = kpom_2012['w'] / (kpom_2012['w'] + kpom_2012['l'] + 0.0)
        kpom_2012['year'] = 2012
        kpom_2012['tour'] = 'ncaa'
        kpom_2012 = kpom_2012.reset_index()
        kpom_all = pd.read_csv("{0}{1}".format(data_prefix,"ncaa_all_kenpom.csv"))
        kpom_all = kpom_all.append(kpom_2012)
        all_results = pd.read_csv("{0}{1}".format(data_prefix,"ncaa_all_results.csv"))
        final_df = generate_final_data(data_prefix, kpom_all, all_results)
        train_fea = pd.DataFrame(index=final_df.index)
        test_fea = pd.DataFrame(index=final_df.index)
        columns = set(final_df.columns)
        train_fea, test_fea =  convert_categorical_to_features(final_df, final_df, columns, train_fea, test_fea)
        del train_fea["SCORE1"]
        del train_fea["SCORE2"]
        targets = train_fea['WINNER']
        print targets
        del train_fea['WINNER']
        print train_fea
        cfr = random_forest_learn(targets, train_fea)
    elif sys.argv[1] == "predict":
        kpom_2013 = pd.read_csv("{0}{1}".format(data_prefix, "2013_ncaa_kpom.csv"), index_col=1)
        rpi_2013 = pd.read_csv("{0}{1}".format(data_prefix, "2013_rpi.csv"), index_col=1)
        for ix in kpom_2013.index:
            if ix not in rpi_2013.index:
                print ix
        kpom_2013 = kpom_2013.join(rpi_2013)
        kpom_2013['w_per'] = kpom_2013['w'] / (kpom_2013['w'] + kpom_2013['l'] + 0.0)
        kpom_2013['year'] = 2013
        kpom_2013['tour'] = 'ncaa'
        kpom_2013 = kpom_2013.reset_index()
        all_2013 = pd.read_csv("{0}{1}".format(data_prefix,"ncaa_2013_results 2nd_round.csv"))
        test_df = generate_final_data(data_prefix, kpom_2013, all_2013)
        
        kpom_2012 = pd.read_csv("{0}{1}".format(data_prefix, "2012_ncaa_kpom.csv"), index_col=1)
        rpi_2012 = pd.read_csv("{0}{1}".format(data_prefix, "2012_rpi.csv"), index_col=1)
        for ix in kpom_2012.index:
            if ix not in rpi_2012.index:
                print ix
        kpom_2012 = kpom_2012.join(rpi_2012)
        kpom_2012['w_per'] = kpom_2012['w'] / (kpom_2012['w'] + kpom_2012['l'] + 0.0)
        kpom_2012['year'] = 2012
        kpom_2012['tour'] = 'ncaa'
        kpom_2012 = kpom_2012.reset_index()
        kpom_all = pd.read_csv("{0}{1}".format(data_prefix,"ncaa_all_kenpom.csv"))
        kpom_all = kpom_all.append(kpom_2012)
        all_results = pd.read_csv("{0}{1}".format(data_prefix,"ncaa_all_results.csv"))
        final_df = generate_final_data(data_prefix, kpom_all, all_results)
        print final_df, len(final_df.columns)
#        print test_df, len(test_df.columns)
        train_fea = pd.DataFrame(index=final_df.index)
        test_fea = pd.DataFrame(index=test_df.index)
        columns = set(final_df.columns)
        train_fea, test_fea =  convert_categorical_to_features(final_df, test_df, columns, train_fea, test_fea)
        del train_fea["SCORE1"]
        del train_fea["SCORE2"]
        targets = train_fea['WINNER']
        targets = targets.apply(lambda x: int(x))
        del train_fea['WINNER']
        cfr = random_forest_learn(targets, train_fea)
        del test_fea["SCORE1"]
        del test_fea["SCORE2"]
        del test_fea["WINNER"]
        for col in train_fea.columns:
            if col not in test_fea.columns:
                print col
#        print train_fea, len(train_fea.columns)
#        print test_fea, len(test_fea.columns)
        expectations, predictions, probs = get_predictions(cfr, test_fea)
        for i, (ix, row) in enumerate(test_df.iterrows()):
#            print "{0} ({1}) vs {2} ({3}) winner {4} with {5}".format(row['TEAM1'], row['SEED1'], row['TEAM2'], row['SEED2'], row['TEAM1'], expectations[i])
            if predictions[i] == 1:
                print "{0} ({1}) vs {2} ({3}) winner {4} with {5}% certainty".format(row['TEAM1'], row['SEED1'], row['TEAM2'], row['SEED2'], row['TEAM1'], int(probs[i][0]*100))
            else:
                print "{0} ({1}) vs {2} ({3}) winner {4} with {5}% certainty".format(row['TEAM1'], row['SEED1'], row['TEAM2'], row['SEED2'], row['TEAM1'], int(probs[i][1]*100))

        
