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
            print e
    else:
        return np.nan
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
def process_flight_history_data(kind, do_header, df, biggest, ignored_columns, header, output_file, clazz=None, ):
    num = 0
    # build up a list for each row of the rows that come before it in time, we are going to flatten those
    unique_cols =  {}
    series = df['actual_gate_departure'].dropna()
    df = df.ix[series.index]
    k = 0
    row_cache = {}
    MAX_NUMBER = 50
    for i in random.sample(series.index, biggest):
        data = ""
        print "working on {0}/{1}".format(k, biggest)
        initial_gate_departure = df.ix[i]['actual_gate_departure']
        if df.ix[i]['actual_gate_departure'] is not None:
            df_tmp = df[df['actual_gate_departure'] < df.ix[i]['actual_gate_departure']]
        else:
            continue
        df_tmp = df_tmp.sort_index(by='actual_gate_departure')
        svm_row = []
        j = 1
        # append class first
        if kind == "svm" and clazz is not None:
            svm_row.append("{0}".format(clazz))
        # loop through all the rows in the previous flights
        for ii, row in enumerate(df_tmp.values[0:MAX_NUMBER]):
            cache = False
            if ii in row_cache:
                svm_row = row_cache[ii]
                cache = True
            for n, val in enumerate(row):
                if not cache:
                    if df.columns[n] not in ignored_columns and str(val) != "nan" and val is not None:
                        if type(val) is datetime.datetime:
                            if k == 0 and kind == "bayesian" and do_header:
                                header += ["{0}_weekday".format(j), "{0}_day".format(j), "{0}_hour".format(j), "{0}_minute".format(j), "{0}_second".format(j)]
                            if kind == "svm":
                                svm_row.append("{0}:{1}".format(j, val.weekday()))
                                j += 1
                            else:
                                svm_row.append("{0}".format(val.weekday()))
                            if kind == "svm":
                                svm_row.append("{0}:{1}".format(j, val.day))
                                j += 1
                            else:
                                svm_row.append("{0}".format(val.day))
                            if kind == "svm":
                                svm_row.append("{0}:{1}".format(j, val.hour))
                                j += 1
                            else:
                                svm_row.append("{0}".format(val.hour))
                            if kind == "svm":
                                svm_row.append("{0}:{1}".format(j, val.minute))
                                j += 1
                            else:
                                svm_row.append("{0}".format(val.minute))
                            if kind == "svm":
                                svm_row.append("{0}:{1}".format(j, val.second))
                                j += 1
                            else:
                                svm_row.append("{0}".format(val.second))
                        else:
                            if k == 0 and kind == "bayesian" and do_header:
                                header.append("{0}_{1}".format(j, df.columns[n]))
                            if kind == "svm":
                                val_tmp = val 
                                if df.dtypes[n] == "object":
                                    # if the column has not been seen before
                                    if df.columns[n] not in unique_cols:
                                        # add it to the unique cols map
                                        unique_cols[df.columns[n]] = []
                                    # if we have not seen this val before
                                    if val not in unique_cols[df.columns[n]]:
                                        # append to the unqiue_cols for this column
                                        unique_cols[df.columns[n]].append(val)
                                        # index is what we want to record for svm (svm uses floats not categorical data (strings))
                                        val_tmp = unique_cols[df.columns[n]].index(val)
                                    else:
                                        # otherwise we get the j for this val in this column
                                        val_tmp = unique_cols[df.columns[n]].index(val)
                                svm_row.append("{0}:{1}".format(j, val_tmp))
                                j += 1
                            else:
                                svm_row.append("{0}".format(val))
                        row_cache[ii] = svm_row 
                    elif k == 0 and kind == "bayesian" and do_header and df.columns[n] not in ignored_columns:
                        header.append("{0}_{1}".format(j, df.columns[n]))
                if df.columns[n] == 'actual_gate_departure':
                    if kind == "bayesian" and do_header:
                        header.append("{0}_gate_time_difference".format(j))
                    diff = minutes_difference(initial_gate_departure, val)
                    if kind == "svm":
                        svm_row.append("{0}:{1}".format(j, diff))
                        j += 1
                    else:
                        svm_row.append("{0}".format(diff))
            if kind == "bayesian":
                j += 1

#                    print val

        
        
        num += 1
        if k == 0 and kind == "bayesian" and do_header:
            data += "%" + ",".join(header) + "\n"
        if kind == "svm":
            data += " ".join(svm_row) + "\n"
        else:
            data += ",".join(svm_row) + "\n"
        k += 1
        output_file.write(data)
    
    return num

def process_flight_history_file(kind, filename, output_file_name, do_header=False):
    df = pd.read_csv('{0}{1}/2012_11_12/FlightHistory/flighthistory.csv'.format(data_prefix, data_rev_prefix), index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=["MISSING"])
    runway_arrival_difference =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
    df['runway_arrival_differences'] = runway_arrival_difference
    df_late = df[df['runway_arrival_differences'] > datetime.timedelta(0, 0, 0)]
    df_ontime = df[df['runway_arrival_differences'] <= datetime.timedelta(0, 0, 0)]
    biggest = len(df_late.values)
    if len(df_ontime.values) < biggest:
        biggest = len(df_ontime.values)
    header = []
    output_file = open(output_file_name, 'w')
    ignored_columns = []
    if kind == "svm":
#        ignored_columns = ["actual_gate_departure", "actual_gate_arrival", "actual_runway_departure", "actual_runway_arrival", "actual_aircraft_type", "runway_arrival_differences"]
        num_negative = process_flight_history_data(kind, do_header, df_late, biggest, ignored_columns, header, output_file, clazz="-1" )
        num_positive = process_flight_history_data(kind, False, df_ontime, biggest, ignored_columns, header, output_file, clazz="+1")
    else:
#        ignored_columns = ["actual_gate_departure", "actual_gate_arrival", "actual_runway_departure", "actual_runway_arrival", "actual_aircraft_type"]
        num_negative = process_flight_history_data(kind, do_header, df_late, biggest, ignored_columns, header, output_file)
        num_positive = process_flight_history_data(kind, False, df_ontime, biggest, ignored_columns, header, output_file)
    output_file.close()
    return num_negative, num_positive

def process_flight_history_file_proxy(args):
    kind = args[0]
    filename = args[1]
    do_header = args[3]
    output_file_name = args[2]
    return process_flight_history_file(kind, filename, output_file_name, do_header)
 
if __name__ == '__main__':
    kind = sys.argv[1]
    num_positive = 0
    num_negative = 0
    i = 0
    pool_queue = []
    pool = Pool(processes=8)
    for subdirname in os.walk('{0}{1}'.format(data_prefix, data_rev_prefix)).next()[1]:
        path = os.path.join('{0}{1}'.format(data_prefix, data_rev_prefix), subdirname)
        print "working on {0}".format('{0}/FlightHistory/flighthistory.csv'.format(path))
        output_file_name = subdirname + "_" + sys.argv[2] + ".tab"
        if kind == "bayesian":
            output_file_name = subdirname + sys.argv[2] + ".csv"
        if i == 0:
            pool_queue.append([kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, True])
#            data, num_negative_tmp, num_postive_tmp = process_flight_history_file(kind, '{0}/FlightHistory/flighthistory.csv'.format(path),output_file_name,  True)
        else:
            pool_queue.append([kind, '{0}/FlightHistory/flighthistory.csv'.format(path),  output_file_name, True])
#            data, num_negative_tmp, num_postive_tmp = process_flight_history_file(kind, '{0}/FlightHistory/flighthistory.csv'.format(path), output_file_name, False)
        i += 1
    result = pool.map(process_flight_history_file_proxy, pool_queue, 1)
    for num_negative_tmp, num_positive_tmp in result:
        num_positive += num_positive_tmp
        num_negative += num_negative_tmp
    print "num_positive: {0}, num_negative: {1}".format(num_positive, num_negative)
