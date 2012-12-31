'''
Created on Dec 17, 2012

@author: jostheim
'''
import pandas as pd
import numpy as np
import pandas.io.date_converters as conv
import datetime
import dateutil
import os
import random
import sys
from multiprocessing import Pool
import pickle


data_prefix = '/Users/jostheim/workspace/kaggle/data/flight_quest/'
data_rev_prefix = 'InitialTrainingSet_rev1'
data_test_rev_prefix = 'SampleTestSet'
#data_transform_dict = {'published_departure':np.float64}

def minutes_difference(datetime1, datetime2):
    diff = datetime1 - datetime2
    return diff.days*24*60+diff.seconds/60

def parse_date_time(val):
    if str(val).lower().strip() != "MISSING" and str(val).lower().strip() != "nan":
        #'2012-11-12 17:30:00+00:00
        try:
            return dateutil.parser.parse(val)
        except ValueError as e:
#            print e
            pass
    else:
        return None
#        return datetime.strptime(val, "%Y-%m-%dT%H:%M:%S")
    

#0 airline_code                         25413  non-null values
#1 airline_icao_code                    25335  non-null values
#2 flight_number                        25413  non-null values
#3 departure_airport_code               25413  non-null values
#4 departure_airport_icao_code          25413  non-null values
#5 arrival_airport_code                 25413  non-null values
#6 arrival_airport_icao_code            25413  non-null values
#7 published_departure                  22258  non-null values
#8 published_arrival                    22258  non-null values
#9 scheduled_gate_departure             22445  non-null values
#10 actual_gate_departure                21113  non-null values
#11 scheduled_gate_arrival               22447  non-null values
#12 *actual_gate_arrival                  21204  non-null values
#13 scheduled_runway_departure           25025  non-null values
#14 actual_runway_departure              23941  non-null values
#15 scheduled_runway_arrival             25020  non-null values
#16 *actual_runway_arrival                23959  non-null values
#17 creator_code                         25413  non-null values
#18 scheduled_air_time                   25020  non-null values
#19 scheduled_block_time                 22443  non-null values
#20 departure_airport_timezone_offset    25413  non-null values
#21 arrival_airport_timezone_offset      25413  non-null values
#22 scheduled_aircraft_type              22258  non-null values
#23 actual_aircraft_type                 0  non-null values
#24 icao_aircraft_type_actual            23692  non-null values
def process_flight_history_data(kind, do_header, df, ignored_columns, header, output_file, unique_cols):
    series = df['actual_runway_arrival'].dropna()
    diffs =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
    df['runway_arrival_diff'] = diffs
    diffs_gate = df['actual_gate_arrival'] - df['scheduled_gate_arrival']
    df['gate_arrival_diff'] = diffs_gate
    if "bayesian" in kind:
        biggest = None
    num = process_flight_history_each(kind, do_header, df, series, None, ignored_columns, header, output_file, unique_cols)
    return num


def process_row(kind, do_header, df, ignored_columns, header, unique_cols, line_count, row_cache, row_count, row, cache, offset, initial_departure=None):
    new_row = []
    column_count = offset
    for column, val in enumerate(row):
        if not cache and df.columns[column] not in ignored_columns:
            if val is not np.nan and val is not None:
                if type(val) is datetime.datetime:
                    if line_count == 0 and "bayesian" in kind and do_header:
                        header += ["{0}_weekday".format(column_count), "{0}_day".format(column_count), "{0}_hour".format(column_count), "{0}_minute".format(column_count), "{0}_second".format(column_count)]
                    if "svm" in kind:
                        new_row.append("{0}:{1}".format(column_count, val.weekday()))
                        column_count += 1
                    else:
                        new_row.append("{0}_{1}:{2}".format(column_count, df.columns[column], val.weekday()))
                    if "svm" in kind:
                        new_row.append("{0}:{1}".format(column_count, val.day))
                        column_count += 1
                    else:
                        new_row.append("{0}_{1}:{2}".format(column_count, df.columns[column], val.day))
                    if "svm" in kind:
                        new_row.append("{0}:{1}".format(column_count, val.hour))
                        column_count += 1
                    else:
                        new_row.append("{0}_{1}:{2}".format(column_count, df.columns[column], val.hour))
                    if "svm" in kind:
                        new_row.append("{0}:{1}".format(column_count, val.minute))
                        column_count += 1
                    else:
                        new_row.append("{0}_{1}:{2}".format(column_count, df.columns[column], val.minute))
                    if "svm" in kind:
                        new_row.append("{0}:{1}".format(column_count, val.second))
                        column_count += 1
                    else:
                        new_row.append("{0}_{1}:{2}".format(column_count, df.columns[column], val.second))
                else:
                    if line_count == 0 and "bayesian" in kind and do_header:
                        header.append("{0}_{1}".format(column_count, df.columns[column]))
                    if "svm" in kind:
                        val_tmp = val
                        if df.dtypes[column] == "object":
                            val_tmp = unique_columns[df.columns[column]].index(val) # otherwise we get the column_count for this val in this column
                        new_row.append("{0}:{1}".format(column_count, val_tmp))
                        column_count += 1
                    else:
                        new_row.append("{0}_{1}:{2}".format(column_count, df.columns[column], val))
#                if row_cache is not None and row_count is not None:
#                    row_cache[row_count] = new_row
            elif line_count == 0 and "bayesian" in kind and do_header and df.columns[column] not in ignored_columns:
                header.append("{0}_{1}".format(column_count, df.columns[column])) # This section builds for each of the flights earlier than the one we are looking at
        # how much earlier it is
        if df.columns[column] == 'scheduled_runway_departure' and initial_departure is not None:
            if "bayesian" in kind and do_header:
                header.append("{0}_runway_time_difference".format(column_count))
            diff = minutes_difference(initial_departure, val)
            if "svm" in kind:
                new_row.append("{0}:{1}".format(column_count, diff))
                column_count += 1
            else:
                new_row.append("{0}_{1}:{2}".format(column_count, "runway_time_difference", diff))
    if "bayesian" in kind:
        column_count += 1
    return new_row, column_count

def process_flight_history_each(kind, do_header, df, series, biggest, ignored_columns, header, output_file, unique_cols):    
    num = 0
    # build up a list for each row of the rows that come before it in time, we are going to flatten those
    offsets_seen = []
    df = df.ix[series.index]
    line_count = 0
    row_cache = {}
    MAX_NUMBER = 50
#    features_frame = pd.DataFrame()
    if biggest is None:
        biggest = len(series)
    for i in random.sample(series.index, biggest):
        data = ""
        svm_row = []
        column_count = 1
        print "working on {0}/{1}".format(line_count, biggest)
        # set the class for svm, we are using multi-class binned by 1 minute
        
        if "svm" in kind and df.ix[i]['runway_arrival_diff'] is np.nan:
            line_count += 1
            continue
        
        runway_arrival_diff = np.nan
        gate_arrival_diff = np.nan
        if df.ix[i]['runway_arrival_diff'] is not np.nan:
            runway_arrival_diff = int(df.ix[i]['runway_arrival_diff'].days*24*60+df.ix[i]['runway_arrival_diff'].seconds/60)
        if df.ix[i]['gate_arrival_diff'] is not np.nan:
            gate_arrival_diff = int(df.ix[i]['gate_arrival_diff'].days*24*60+df.ix[i]['gate_arrival_diff'].seconds/60)
        
        if "svm" in kind: 
            if runway_arrival_diff > 0:
                if "svm" in kind:
                    svm_row.append(str(runway_arrival_diff))
            else:
                line_count += 1
                continue
        
        if "bayesian" in kind:
            if line_count == 0:
                header.append("runway_arrival_diff")
                header.append("gate_arrival_diff")
            if runway_arrival_diff is not np.nan:
                svm_row.append("{0}:{1}".format("runway_arrival_diff", runway_arrival_diff))
            if gate_arrival_diff is not np.nan:
                svm_row.append("{0}:{1}".format("gate_arrival_diff", gate_arrival_diff))

        # we want to create a new data frame with all the flights with scheduled departures less than
        # this flights, these are flights that left before this flight and therefore could be
        # correlated with this flights arrival
        # using scheduled b/c it should be availble all the time
        if df.ix[i]['scheduled_runway_departure'] is not None:
            df_tmp = df[df['scheduled_runway_departure'] < df.ix[i]['scheduled_runway_departure']]
            df_tmp['scheduled_runway_diff'] = df.ix[i]['scheduled_runway_departure'] - df['scheduled_runway_departure'] 
        else:
            line_count += 1
            continue
        # this flights data
        new_row, column_count = process_row(kind, do_header, df, ignored_columns, header, unique_cols, line_count, None, None, df.ix[i], False, 1, unique_cols)
        svm_row += new_row
        # this is the scheduled gate departure for the current flight (flight index i)
        initial_departure = df.ix[i]['scheduled_runway_departure']
        # sort the flights with scheduled departures less than this ones, so we are always in some sort of order
        df_tmp = df_tmp.sort_index(by='scheduled_runway_departure')
        # loop through all the rows in the previous flights
        for row_count, row in enumerate(df_tmp.values[0:MAX_NUMBER]):
            cache = False
            offset = column_count# int((row[len(row)-1].days*24*60+row[len(row)-1].seconds/60))*len(df.columns)
            if row_count in row_cache:
#               svm_row += row_cache[row_count]
#               cache = True
                pass
            else:
#                if offset not in offsets_seen:

                    new_row, column_count = process_row(kind, do_header, df_tmp, ignored_columns, header, unique_cols, line_count, row_cache, row_count, row, cache, offset, initial_departure=initial_departure)
                    svm_row += new_row
#                    offsets_seen.append(offset)
        num += 1
        if line_count == 0 and "bayesian" in kind and do_header:
            data += "%" + ",".join(header) + "\n"
        if "svm" in kind:
            data += " ".join(svm_row) + "\n"
        else:
            data += ",".join(svm_row) + "\n"
        line_count += 1
        output_file.write(data)
    
    return num

def process_flight_history_file(kind, filename, output_file_name, test_filename, unique_cols):
    df = pd.read_csv(filename, index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
#    df_test = pd.read_csv(test_filename, index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
    # still confused, but we may want to remove all data in the test set (df_test) from the training set (df)
    header = []
    ignored_columns = ['actual_gate_arrival','actual_runway_arrival','runway_arrival_diff', 'runway_arrival_differences' ]
    output_file = open(output_file_name, 'w')
#   ignored_columns = ["actual_gate_departure", "actual_gate_arrival", "actual_runway_departure", "actual_runway_arrival", "actual_aircraft_type", "runway_arrival_differences"]
    num = process_flight_history_data(kind, "bayesian" in kind, df, ignored_columns, header, output_file, unique_cols)
    output_file.close()
    return num

def get_base_data():
    unique_cols = {}
    runway_arrival_diff = None
    gate_arrival_diff = None
    for subdirname in os.walk('{0}{1}'.format(data_prefix, data_rev_prefix)).next()[1]:
        path = os.path.join('{0}{1}'.format(data_prefix, data_rev_prefix), subdirname)
        print "working on {0}".format('{0}/FlightHistory/flighthistory.csv'.format(path))
        filename = '{0}/FlightHistory/flighthistory.csv'.format(path)
        df = pd.read_csv(filename, index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
        
        diffs =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
        df['runway_arrival_diff'] = diffs
        diffs_gate = df['actual_gate_arrival'] - df['scheduled_gate_arrival']
        series = diffs.dropna()
        all_diffs = []
        for diff in series:
            all_diffs.append(diff.days*24*60+diff.seconds/60)
        all_diffs = np.asarray(all_diffs)
        if runway_arrival_diff is None:
            runway_arrival_diff = [np.max(all_diffs), np.min(all_diffs)]
        else:
            if np.max(all_diffs) >  runway_arrival_diff[0]:
                runway_arrival_diff[0] = np.max(all_diffs)
            if np.min(all_diffs) < runway_arrival_diff[1]:
                runway_arrival_diff[1] = np.min(all_diffs)
        series = diffs_gate.dropna()
        all_diffs = []
        for diff in series:
            all_diffs.append(diff.days*24*60+diff.seconds/60)
        all_diffs = np.asarray(all_diffs)
        if gate_arrival_diff is None:
            gate_arrival_diff = [np.max(all_diffs), np.min(all_diffs)]
        else:
            if np.max(all_diffs) >  gate_arrival_diff[0]:
                gate_arrival_diff[0] = np.max(all_diffs)
            if np.min(all_diffs) < gate_arrival_diff[1]:
                gate_arrival_diff[1] = np.min(all_diffs)
                
        for row in df.values:
            for column, val in enumerate(row):
                if df.dtypes[column] == "object" and type(val) is datetime.datetime:
                    # if the column has not been seen before
                    if df.columns[column] not in unique_cols:
                        # add it to the unique cols map
                        unique_cols[df.columns[column]] = [] # if we have not seen this val before
                    if val not in unique_cols[df.columns[column]]: # append to the unqiue_cols for this column
                        unique_cols[df.columns[column]].append(val) # index is what we want to record for svm (svm uses floats not categorical data (strings))
        
    dict = {"gate_arrival_min":gate_arrival_diff[1], "gate_arrival_max":gate_arrival_diff[0], "runway_arrival_min":runway_arrival_diff[1], "runway_arrival_max":runway_arrival_diff[0]}
    pickle.dump(dict, open("min_maxes.pickle", "wb"))
    
    pickle.dump(unique_cols, open("unique_columns.pickle", "wb"))
    

def process_flight_history_file_proxy(args):
    kind = args[0]
    filename = args[1]
    output_file_name = args[2]
    test_file_name = args[3]
    unique_cols = args[4]
    return process_flight_history_file(kind, filename, output_file_name, test_file_name, unique_cols)
 
if __name__ == '__main__':
    kind = sys.argv[1]
    num = 0
    i = 0
    unique_columns = None
    if "svm" in kind:
        try:
            unique_columns = pickle.load(open("unique_columns.p",'rb'))
            min_maxes = pickle.load(open("min_maxes.p", 'rb'))
        except Exception as e:
            get_base_data()
            unique_columns = pickle.load(open("unique_columns.p", 'rb'))
            min_maxes = pickle.load(open("min_maxes.p", 'rb'))
    pool_queue = []
    pool = Pool(processes=8)
    for subdirname in os.walk('{0}{1}'.format(data_prefix, data_rev_prefix)).next()[1]:
        path = os.path.join('{0}{1}'.format(data_prefix, data_rev_prefix), subdirname)
        test_path = os.path.join('{0}{1}'.format(data_prefix, data_test_rev_prefix), subdirname)
        print "working on {0}".format('{0}/FlightHistory/flighthistory.csv'.format(path))
        output_file_name = subdirname + "_" + sys.argv[2] + ".tab"
        if "bayesian" in kind:
            output_file_name = subdirname + sys.argv[2] + ".csv"
        if i == 0:
            if 'multi' in kind:
                pool_queue.append([kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), unique_columns])
            else:
                num_tmp = process_flight_history_file(kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), unique_columns)
        else:
            if 'multi' in kind:
                pool_queue.append([kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), unique_columns])
            else:
                num_tmp, num_postive_tmp = process_flight_history_file(kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), unique_columns)
        i += 1
    result = pool.map(process_flight_history_file_proxy, pool_queue, 1)
    for num_tmp in result:
        num += num_tmp
    print "num: {0}".format(num)
