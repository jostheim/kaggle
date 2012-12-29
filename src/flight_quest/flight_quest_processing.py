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
def process_flight_history_data(kind, do_header, df, ignored_columns, header, output_file):
    series = df['actual_runway_arrival'].dropna()
    diffs =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
    df['runway_arrival_diff'] = diffs
    if "bayesian" in kind:
        biggest = None
    num = process_flight_history_each(kind, do_header, df, series, None, ignored_columns, header, output_file)
    return num


def process_row(kind, do_header, df, ignored_columns, header, unique_cols, line_count, row_cache, svm_row, column_count, row_count, row, cache, initial_gate_departure=None):
    for column, val in enumerate(row):
        if not cache and df.columns[column] not in ignored_columns:
            if val is not np.nan and str(val) != "nan" and val is not None:
                if type(val) is datetime.datetime:
                    if line_count == 0 and "bayesian" in kind and do_header:
                        header += ["{0}_weekday".format(column_count), "{0}_day".format(column_count), "{0}_hour".format(column_count), "{0}_minute".format(column_count), "{0}_second".format(column_count)]
                    if "svm" in kind:
                        svm_row.append("{0}:{1}".format(column_count, val.weekday()))
                        column_count += 1
                    else:
                        svm_row.append("{0}".format(val.weekday()))
                    if "svm" in kind:
                        svm_row.append("{0}:{1}".format(column_count, val.day))
                        column_count += 1
                    else:
                        svm_row.append("{0}".format(val.day))
                    if "svm" in kind:
                        svm_row.append("{0}:{1}".format(column_count, val.hour))
                        column_count += 1
                    else:
                        svm_row.append("{0}".format(val.hour))
                    if "svm" in kind:
                        svm_row.append("{0}:{1}".format(column_count, val.minute))
                        column_count += 1
                    else:
                        svm_row.append("{0}".format(val.minute))
                    if "svm" in kind:
                        svm_row.append("{0}:{1}".format(column_count, val.second))
                        column_count += 1
                    else:
                        svm_row.append("{0}".format(val.second))
                else:
                    if line_count == 0 and "bayesian" in kind and do_header:
                        header.append("{0}_{1}".format(column_count, df.columns[column]))
                    if "svm" in kind:
                        val_tmp = val
                        if df.dtypes[column] == "object":
                            # if the column has not been seen before
                            if df.columns[column] not in unique_cols:
                                # add it to the unique cols map
                                unique_cols[df.columns[column]] = [] # if we have not seen this val before
                            if val not in unique_cols[df.columns[column]]: # append to the unqiue_cols for this column
                                unique_cols[df.columns[column]].append(val) # index is what we want to record for svm (svm uses floats not categorical data (strings))
                                val_tmp = unique_cols[df.columns[column]].index(val)
                            else:
                                val_tmp = unique_cols[df.columns[column]].index(val) # otherwise we get the column_count for this val in this column
                        svm_row.append("{0}:{1}".format(column_count, val_tmp))
                        column_count += 1
                    else:
                        svm_row.append("{0}".format(val))
                if row_count is not None:
                    row_cache[row_count] = svm_row
            elif line_count == 0 and "bayesian" in kind and do_header and df.columns[column] not in ignored_columns:
                header.append("{0}_{1}".format(column_count, df.columns[column])) # This section builds for each of the flights earlier than the one we are looking at
        # how much earlier it is
        if df.columns[column] == 'scheduled_gate_departure' and initial_gate_departure is not None:
            if "bayesian" in kind and do_header:
                header.append("{0}_gate_time_difference".format(column_count))
            diff = minutes_difference(initial_gate_departure, val)
            if "svm" in kind:
                svm_row.append("{0}:{1}".format(column_count, diff))
                column_count += 1
            else:
                svm_row.append("{0}".format(diff))
    if "bayesian" in kind:
        column_count += 1
    return column_count

def process_flight_history_each(kind, do_header, df, series, biggest, ignored_columns, header, output_file):    
    num = 0
    # build up a list for each row of the rows that come before it in time, we are going to flatten those
    unique_cols =  {}
    df = df.ix[series.index]
    line_count = 0
    row_cache = {}
    MAX_NUMBER = 50
    if biggest is None:
        biggest = len(series)
    for i in random.sample(series.index, biggest):
        data = ""
        svm_row = []
        column_count = 1
        print "working on {0}/{1}".format(line_count, biggest)
        # set the class for svm, we are using multi-class binned by 1 minute
        if "svm" in kind:
            if df.ix[i]['runway_arrival_diff'] is np.nan:
                continue
            diff = int(df.ix[i]['runway_arrival_diff'].days*24*60+df.ix[i]['runway_arrival_diff'].seconds/60)
            if diff > 0:
                svm_row.append(str(diff))
            else:
                continue
        # this flights data
        column_count = process_row(kind, do_header, df, ignored_columns, header, unique_cols, line_count, row_cache, svm_row, column_count, 0, df.ix[i], False)
        # this is the scheduled gate departure for the current flight (flight index i)
        initial_gate_departure = df.ix[i]['scheduled_gate_departure']
        # we want to create a new data frame with all the flights with scheduled departures less than
        # this flights, these are flights that left before this flight and therefore could be
        # correlated with this flights arrival
        # using scheduled b/c it should be availble all the time
        if df.ix[i]['scheduled_gate_departure'] is not None:
            df_tmp = df[df['scheduled_gate_departure'] < df.ix[i]['scheduled_gate_departure']]
        else:
            continue
        # sort the flights with scheduled departures less than this ones, so we are always in some sort of order
        df_tmp = df_tmp.sort_index(by='scheduled_gate_departure')
        # loop through all the rows in the previous flights
        for row_count, row in enumerate(df_tmp.values[0:MAX_NUMBER]):
            cache = False
            if row_count in row_cache:
                svm_row = row_cache[row_count]
                cache = True
            column_count = process_row(kind, do_header, df, ignored_columns, header, unique_cols, line_count, row_cache, svm_row, column_count, row_count, row, cache, initial_gate_departure=initial_gate_departure)
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

def process_flight_history_file(kind, filename, output_file_name, test_filename, do_header=False):
    df = pd.read_csv(filename, index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
#    df_test = pd.read_csv(test_filename, index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
    # still confused, but we may want to remove all data in the test set (df_test) from the training set (df)
    
    runway_arrival_difference =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
    df['runway_arrival_differences'] = runway_arrival_difference
    header = []
    ignored_columns = ['actual_gate_arrival','actual_runway_arrival','runway_arrival_diff', 'runway_arrival_differences' ]
    output_file = open(output_file_name, 'w')
#   ignored_columns = ["actual_gate_departure", "actual_gate_arrival", "actual_runway_departure", "actual_runway_arrival", "actual_aircraft_type", "runway_arrival_differences"]
    num = process_flight_history_data(kind, do_header, df, ignored_columns, kind == "bayesian", output_file)
    output_file.close()
    return num

def get_maxes_and_mins(filename):
    df = pd.read_csv('{0}{1}/2012_11_12/FlightHistory/flighthistory.csv'.format(data_prefix, data_rev_prefix), index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
    diffs =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
    series = diffs.dropna()
    all_diffs = []
    for diff in series:
        all_diffs.append(diff.days*24*60+diff.seconds/60)
    all_diffs = np.asarray(all_diffs)
    return np.max(all_diffs), np.min(all_diffs)
    

def process_flight_history_file_proxy(args):
    kind = args[0]
    filename = args[1]
    do_header = args[3]
    output_file_name = args[2]
    return process_flight_history_file(kind, filename, output_file_name, do_header)
 
if __name__ == '__main__':
    kind = sys.argv[1]
    num = 0
    i = 0
    if kind == 'max_min':
        most_max = None
        most_min = None
        for subdirname in os.walk('{0}{1}'.format(data_prefix, data_rev_prefix)).next()[1]:
            path = os.path.join('{0}{1}'.format(data_prefix, data_rev_prefix), subdirname)
            print "working on {0}".format('{0}/FlightHistory/flighthistory.csv'.format(path))
            max, min = get_maxes_and_mins('{0}/FlightHistory/flighthistory.csv'.format(path))
            if most_max is None or max > most_max:
                most_max = max
            if most_min is None or (min < most_min and min > 0):
                most_min = min
                if most_min < 0:
                    most_min = 0
        print most_max, most_min
    else:
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
                    pool_queue.append([kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), True])
                else:
                    num_negative_tmp, num_postive_tmp = process_flight_history_file(kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), True)
            else:
                if 'multi' in kind:
                    pool_queue.append([kind, '{0}/FlightHistory/flighthistory.csv'.format(path), '{0}/FlightHistory/flighthistory.csv'.format(test_path), output_file_name, True])
                else:
                    num_negative_tmp, num_postive_tmp = process_flight_history_file(kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, '{0}/FlightHistory/flighthistory.csv'.format(test_path), False)
            i += 1
        result = pool.map(process_flight_history_file_proxy, pool_queue, 1)
        for num_tmp in result:
            num += num_tmp
        print "num: {0}".format(num)
