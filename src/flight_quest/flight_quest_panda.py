'''
Created on Jan 6, 2013

@author: jostheim
'''
import pandas as pd
import numpy as np
import types
from flight_quest_processing import parse_date_time
import datetime
from multiprocessing import Pool
import os, sys

data_prefix = '/Users/jostheim/workspace/kaggle/data/flight_quest/'
data_rev_prefix = 'InitialTrainingSet_rev1'
date_prefix = '2012_11_12'
data_test_rev_prefix = 'SampleTestSet'
na_values = ["MISSING", "HIDDEN"]

def get_flight_history():
    filename = "{0}/{1}/{2}/FlightHistory/flighthistory.csv".format(data_prefix, data_rev_prefix, date_prefix)
    df = pd.read_csv(filename, index_col=0, parse_dates=[7,8,9,10,11,12,13,14,15,16,17], date_parser=parse_date_time, na_values=na_values)
    return df

def get_metar(prefix):
    filename = "{0}/{1}/{2}/metar/flightstats_metarreports_combined.csv".format(data_prefix, data_rev_prefix, date_prefix)
    metar_df = pd.read_csv(filename, index_col=['metar_reports_id'], parse_dates=[2], date_parser=parse_date_time, na_values=na_values)
    filename = "{0}/{1}/{2}/metar/flightstats_metarpresentconditions_combined.csv".format(data_prefix, data_rev_prefix, date_prefix)
    metar_present_conditions_df = pd.read_csv(filename, index_col=['metar_reports_id'], na_values=na_values)
    del metar_present_conditions_df['id'] 
    filename = "{0}/{1}/{2}/metar/flightstats_metarrunwaygroups_combined.csv".format(data_prefix, data_rev_prefix, date_prefix)
    metar_runway_groups_df = pd.read_csv(filename, index_col=['metar_reports_id'], na_values=na_values)
    del metar_runway_groups_df['id'] 
    filename = "{0}/{1}/{2}/metar/flightstats_metarskyconditions_combined.csv".format(data_prefix, data_rev_prefix, date_prefix)
    metar_sky_conditions_df = pd.read_csv(filename, index_col=['metar_reports_id'], na_values=na_values)
    del metar_sky_conditions_df['id'] 
    metar_join = metar_present_conditions_df.join([metar_runway_groups_df, metar_sky_conditions_df])
    metar_df = metar_df.join(metar_join)
    
    grouped = metar_df.groupby("weather_station_code")
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'weather_station_code':name}
        group = group.sort_index(by='date_time_issued')
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "weather_station_code":
                    d["{0}_{1}_{2}".format(prefix, k, group.columns[j])] = val
        groups.append(d)
        i += 1
    tmp_df = pd.DataFrame(groups)
    tmp_df.set_index('weather_station_code', inplace=True, verify_integrity=True)
    return tmp_df

def get_fbwind(prefix):
    filename = "{0}/{1}/{2}/otherweather/flightstats_fbwindreport.csv".format(data_prefix, data_rev_prefix, date_prefix)
    fbwindreport_df = pd.read_csv(filename, parse_dates=[1], date_parser=parse_date_time, na_values=na_values)
    filename = "{0}/{1}/{2}/otherweather/flightstats_fbwindairport.csv".format(data_prefix, data_rev_prefix, date_prefix)
    fbwindairport_df = pd.read_csv(filename, na_values=na_values)
    filename = "{0}/{1}/{2}/otherweather/flightstats_fbwind.csv".format(data_prefix, data_rev_prefix, date_prefix)
    fbwind_df = pd.read_csv(filename, na_values=na_values)
    filename = "{0}/{1}/{2}/otherweather/flightstats_fbwindaltitude.csv".format(data_prefix, data_rev_prefix, date_prefix)
    fbwindaltitude_df = pd.read_csv(filename, na_values=na_values)
    grouped = fbwindaltitude_df.groupby('fbwindreportid')
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'fbwindreportid':int(name)}
        group = group.sort_index(by="ordinal")
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "fbwindreportid" and group.columns[j] != "ordinal":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    fbwindaltitude_flattened_df = pd.DataFrame(groups)
    grouped = fbwind_df.groupby('fbwindairportid')
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'fbwindairportid':int(name)}
        group = group.sort_index(by="ordinal")
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "fbwindairportid" and group.columns[j] != "ordinal":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    fbwind_flattened_df = pd.DataFrame(groups)

    fbwind_joined = pd.merge(fbwindairport_df, fbwind_flattened_df, how="left", left_on="fbwindairportid", right_on="fbwindairportid")
    print fbwind_joined.values
    
    fbwindreport_df_joined = pd.merge(fbwindreport_df, fbwindaltitude_flattened_df, how="left", left_on="fbwindreportid", right_on="fbwindreportid")
    fbwindreport_df_joined.set_index("fbwindreportid", inplace=True, verify_integrity=True)
    
    fb_wind_df = pd.merge(fbwind_joined, fbwindreport_df_joined, how="left", left_on='fbwindreportid', right_index=True)
    del fb_wind_df['fbwindreportid']
    del fb_wind_df['fbwindairportid']
    
    grouped = fbwind_df.groupby('airportcode')
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'airportcode':name}
        group = group.sort_index(by="createdutc")
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "airportcode":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    fbwind_df = pd.DataFrame(groups)
    fbwind_df.set_index("airportcode", inplace=True, verify_integrity=True)
    new_col_names = []
    for col in fb_wind_df.columns:
        new_col_names.append("{0}_{1}".format(prefix, col))
    fb_wind_df.columns = new_col_names
    return fb_wind_df

def get_flight_history_events():
    events_filename = "{0}/{1}/{2}/FlightHistory/flighthistoryevents.csv".format(data_prefix, data_rev_prefix, date_prefix)
    events_df = pd.read_csv(events_filename, na_values=na_values, parse_dates=[1], date_parser=parse_date_time)
    grouped = events_df.groupby("flight_history_id")
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'flight_history_id':int(name)}
        group = group.sort_index(by='date_time_recorded')
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "flight_history_id":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    tmp_df = pd.DataFrame(groups)
    tmp_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    return tmp_df

def get_asdi_root(): 
    asdi_root_filename = "{0}/{1}/{2}/ASDI/asdiflightplan.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_root_df = pd.read_csv(asdi_root_filename, na_values=na_values, index_col=['asdiflightplanid'], parse_dates=[1,7,8,9,10], date_parser=parse_date_time)
    return asdi_root_df

def get_asdi_airway():
    asdi_airway_filename = "{0}/{1}/{2}/ASDI/asdiairway.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_airway_df = pd.read_csv(asdi_airway_filename, index_col=['asdiflightplanid','ordinal'], na_values=na_values)
    return asdi_airway_df

def get_asdi_fpcenter():
    asdi_asdifpcenter_filename = "{0}/{1}/{2}/ASDI/asdifpcenter.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_asdifpcenter_df = pd.read_csv(asdi_asdifpcenter_filename, index_col=['asdiflightplanid','ordinal'], na_values=na_values)
    return asdi_asdifpcenter_df

def get_asdi_pfix():
    asdi_asdifpfix_filename = "{0}/{1}/{2}/ASDI/asdifpfix.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_asdifpfix_df = pd.read_csv(asdi_asdifpfix_filename, index_col=['asdiflightplanid','ordinal'], na_values=na_values)
    return asdi_asdifpfix_df

def get_asdi_psector():
    asdi_asdifpsector_filename = "{0}/{1}/{2}/ASDI/asdifpsector.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_asdifpsector_df = pd.read_csv(asdi_asdifpsector_filename, index_col=['asdiflightplanid','ordinal'], na_values=na_values)
    return asdi_asdifpsector_df

def get_asdi_waypoint():
    asdi_asdifpwaypoint_filename = "{0}/{1}/{2}/ASDI/asdifpwaypoint.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_asdifpwaypoint_df = pd.read_csv(asdi_asdifpwaypoint_filename, index_col=['asdiflightplanid','ordinal'], na_values=na_values)
    return asdi_asdifpwaypoint_df

def get_asdi_disposition():
    asdi_asdiposition_filename = "{0}/{1}/{2}/ASDI/asdiposition.csv".format(data_prefix, data_rev_prefix, date_prefix)
    asdi_asdiposition_df = pd.read_csv(asdi_asdiposition_filename, parse_dates=[0], date_parser=parse_date_time, na_values=na_values)
    grouped = asdi_asdiposition_df.groupby('flighthistoryid')
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'flight_history_id':int(name)}
        group = group.sort_index(by='received')
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "flighthistoryid":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    tmp_df = pd.DataFrame(groups)
    tmp_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    return tmp_df

def get_asdi_merged():
    asdi_root_df = get_asdi_root()
    asdi_airway_df = get_asdi_airway()
    asdi_asdifpcenter_df = get_asdi_fpcenter()
    asdi_asdifpfix_df = get_asdi_pfix()
    asdi_asdifpsector_df = get_asdi_psector()
    asdi_asdifpwaypoint_df = get_asdi_waypoint()
    tmp = asdi_airway_df.join([asdi_asdifpcenter_df, asdi_asdifpfix_df, asdi_asdifpsector_df, asdi_asdifpwaypoint_df])
    grouped = tmp.groupby(level='asdiflightplanid')
    groups = []
    i = 0
    
    for name, group in grouped:
        d = {}
    #   print "switching"
        #print group.index[n]
        #group = group.sort_index(by='ordinal')
        for k, row in enumerate(group.values):
            #print row, group.index[k]
            for j, val in enumerate(row):
                if "asdiflightplanid" not in d:
                    d["asdiflightplanid"] = group.index[k][0]
                if group.columns[j] != "asdiflightplanid":
                    d["{0}_{1}".format(group.index[k][1], group.columns[j])] = val
        groups.append(d)
        i += 1
    tmp_df = pd.DataFrame(groups)
    tmp_df.set_index('asdiflightplanid', inplace=True, verify_integrity=True)
    asdi_merged_df = pd.merge(asdi_root_df, tmp_df, how="left", left_index=True, right_index=True)
    return asdi_merged_df

def get_atscc_deicing():
    atsccdeicing_filename = "{0}/{1}/{2}/atscc/flightstats_atsccdeicing.csv".format(data_prefix, data_rev_prefix, date_prefix)
    atsccdeicing_df = pd.read_csv(atsccdeicing_filename, index_col=[0], na_values=na_values, parse_dates=[1,2,3,4], date_parser=parse_date_time)
    end_time = []
    for ix,row in atsccdeicing_df.iterrows():
        end = row['end_time']
        if end is None or (row['invalidated_time'] is not None and row['invalidated_time'] < end):
            end = row['invalidated_time']
        end_time.append(end)
    atsccdeicing_df['actual_end_time'] = pd.Series(end_time, index=atsccdeicing_df.index)
    return atsccdeicing_df

def get_atscc_delay():
    atsccdelay_filename = "{0}/{1}/{2}/atscc/flightstats_atsccdelay.csv".format(data_prefix, data_rev_prefix, date_prefix)
    atsccdelay_df = pd.read_csv(atsccdelay_filename, index_col=[0], na_values=na_values, parse_dates=[1,2,3,4], date_parser=parse_date_time)
    end_time = []
    for ix,row in atsccdelay_df.iterrows():
        end = row['end_time']
        if end is None or (row['invalidated_time'] is not None and row['invalidated_time'] < end):
            end = row['invalidated_time']
        end_time.append(end)
    atsccdelay_df['actual_end_time'] = pd.Series(end_time, index=atsccdelay_df.index)
    return atsccdelay_df

def get_atscc_ground_delay():
    atsccgrounddelay_filename = "{0}/{1}/{2}/atscc/flightstats_atsccgrounddelay.csv".format(data_prefix, data_rev_prefix, date_prefix)
    atsccgrounddelay_df = pd.read_csv(atsccgrounddelay_filename, index_col=[0], na_values=na_values, parse_dates=[1,2,3,4,5,8,9,10,11], date_parser=parse_date_time)
    atsccgrounddelayairports_filename = "{0}/{1}/{2}/atscc/flightstats_atsccgrounddelayairports.csv".format(data_prefix, data_rev_prefix, date_prefix)
    atsccgrounddelayairports_df = pd.read_csv(atsccgrounddelayairports_filename, na_values=na_values)
    grouped = atsccgrounddelayairports_df.groupby('ground_delay_program_id')
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'ground_delay_program_id':int(name)}
        for k, row in enumerate(group.values):
            for j, val in enumerate(row):
                if group.columns[j] != "ground_delay_program_id":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    atsccgrounddelayairports_tmp_df = pd.DataFrame(groups)
    atsccgrounddelayairports_tmp_df.set_index('ground_delay_program_id', inplace=True, verify_integrity=True)
    atsccgrounddelay_merged_df = pd.merge(atsccgrounddelay_df, atsccgrounddelayairports_tmp_df, how="left", left_index=True, right_index=True)
    atsccgrounddelayartccs_filename = "{0}/{1}/{2}/atscc/flightstats_atsccgrounddelayartccs.csv".format(data_prefix, data_rev_prefix, date_prefix)
    atsccgrounddelayartccs_df = pd.read_csv(atsccgrounddelayartccs_filename, na_values=na_values)
    grouped = atsccgrounddelayartccs_df.groupby('ground_delay_program_id')
    groups = []
    i = 0
    for name, group in grouped:
        d = {}
    #   print "switching"
        d = {'ground_delay_program_id':int(name)}
        for k, row in enumerate(group.values):
    #        print row
            for j, val in enumerate(row):
                if group.columns[j] != "ground_delay_program_id":
                    d["{0}_{1}".format(k, group.columns[j])] = val
        groups.append(d)
        i += 1
    atsccgrounddelayartccs_tmp_df = pd.DataFrame(groups)
    atsccgrounddelayartccs_tmp_df.set_index('ground_delay_program_id', inplace=True, verify_integrity=True)
    #atsccgrounddelay_merged_df = pd.merge(atsccgrounddelay_merged_df, atsccgrounddelayartccs_tmp_df, how="left", left_index=True, right_index=True)
    end_time = []
    for ix,row in atsccgrounddelay_merged_df.iterrows():
        end = row['effective_end_time']
        if end is None or (row['invalidated_time'] is not None and row['invalidated_time'] < end):
            end = row['invalidated_time']
        if end is None or (row['cancelled_time'] is not None and row['cancelled_time'] < end):
            end = row['cancelled_time']
        end_time.append(end)
    atsccgrounddelay_merged_df['actual_end_time'] = pd.Series(end_time, index=atsccgrounddelay_merged_df.index)
    return atsccgrounddelay_merged_df

def merge(left, right):
    return pd.merge(left, right, how="left", left_on=['flight_history_id'], right_on=['flighthistoryid'])

def get_for_flights(df):
    atsccgrounddelay_merged_df = get_atscc_ground_delay()
    atsccdelay_df = get_atscc_delay()
    atsccdeicing_df = get_atscc_deicing()
    arrival_ground_delays = []
    departure_ground_delays = []
    arrival_delays = []
    departure_delays = []
    arrival_icing_delays = []
    departure_icing_delays = []
    for ix, row in df.iterrows():
        arrival_grounddelay = {}
        departure_grounddelay = {}
        arrival_delay = {}
        departure_delay = {}
        arrival_icing_delay = {}
        departure_icing_delay = {}
        
        # if our scheduled gate arrival is within a ground delay
        if row['scheduled_gate_arrival'] is not None and row['scheduled_gate_arrival'] is not np.nan:
            tmp = atsccgrounddelay_merged_df[atsccgrounddelay_merged_df['airport_code'] == row['arrival_airport_code']]
            tmp = tmp[tmp['effective_start_time'] <= row['scheduled_gate_arrival']]
            tmp = tmp[tmp['actual_end_time'] >= row['scheduled_gate_arrival']]
            tmp = tmp.sort_index(by='effective_start_time')
            i = 0
            for ix_tmp, row_tmp in tmp.iterrows():
                for j, val in enumerate(row_tmp):
                    arrival_grounddelay["arrival_ground_delay_{0}_{1}".format(i, tmp.columns[j])] = val
                    #print row['scheduled_gate_arrival'], row['departure_airport_code'], row_tmp['effective_start_time'], row_tmp['effective_end_time'], row_tmp['airport_code']
                i += 1
            if len(arrival_grounddelay) > 0:
                arrival_grounddelay['flight_history_id'] = ix
                arrival_ground_delays.append(arrival_grounddelay)
                
            tmp = atsccdelay_df[atsccdelay_df['airport_code'] == row['arrival_airport_code']]
            tmp = tmp[tmp['start_time'] <= row['scheduled_gate_arrival']]
            tmp = tmp[tmp['actual_end_time'] >= row['scheduled_gate_arrival']]
            tmp = tmp.sort_index(by='start_time')
            i = 0
            for ix_tmp, row_tmp in tmp.iterrows():
                for j, val in enumerate(row_tmp):
                    arrival_delay["arrival_delay_{0}_{1}".format(i, tmp.columns[j])] = val
                i += 1
            if len(arrival_delay) > 0:
                arrival_delay['flight_history_id'] = ix
                arrival_delays.append(arrival_delay)
            
            tmp = atsccdeicing_df[atsccdeicing_df['airport_code'] == row['arrival_airport_code']]
            tmp = tmp[tmp['start_time'] <= row['scheduled_gate_arrival']]
            tmp = tmp[tmp['actual_end_time'] >= row['scheduled_gate_arrival']]
            tmp = tmp.sort_index(by='start_time')
            i = 0
            for ix_tmp, row_tmp in tmp.iterrows():
                for j, val in enumerate(row_tmp):
                    arrival_icing_delay["arrival_icing_delay_{0}_{1}".format(i, tmp.columns[j])] = val
                i += 1
            if len(arrival_icing_delay) > 0:
                arrival_icing_delay['flight_history_id'] = ix
                arrival_icing_delays.append(arrival_icing_delay)
                
        if row['scheduled_gate_departure'] is not None and row['scheduled_gate_departure'] is not np.nan:
            tmp = atsccgrounddelay_merged_df[atsccgrounddelay_merged_df['airport_code'] == row['departure_airport_code']]
            tmp = tmp[tmp['effective_start_time'] <= row['scheduled_gate_departure']]
            tmp = tmp[tmp['actual_end_time'] >= row['scheduled_gate_departure']]
            tmp = tmp.sort_index(by='effective_start_time')
            i = 0
            for ix_tmp, row_tmp in tmp.iterrows():
                for j, val in enumerate(row_tmp):
                    departure_grounddelay["departure_ground_delay_{0}_{1}".format(i, tmp.columns[j])] = val
                    #print row['scheduled_gate_arrival'], row['departure_airport_code'], row_tmp['effective_start_time'], row_tmp['effective_end_time'], row_tmp['airport_code']
                i += 1
            if len(departure_grounddelay) > 0:
                departure_grounddelay['flight_history_id'] = ix
                departure_ground_delays.append(departure_grounddelay)
                #print departure_grounddelay
    
            tmp = atsccdelay_df[atsccdelay_df['airport_code'] == row['departure_airport_code']]
            tmp = tmp[tmp['start_time'] <= row['scheduled_gate_departure']]
            tmp = tmp[tmp['actual_end_time'] >= row['scheduled_gate_departure']]
            tmp = tmp.sort_index(by='start_time')
            i = 0
            for ix_tmp, row_tmp in tmp.iterrows():
                for j, val in enumerate(row_tmp):
                    departure_delay["departure_delay_{0}_{1}".format(i, tmp.columns[j])] = val
                i += 1
            if len(departure_delay) > 0:
                departure_delay['flight_history_id'] = ix
                departure_delays.append(departure_delay)
            
            tmp = atsccdeicing_df[atsccdeicing_df['airport_code'] == row['departure_airport_code']]
            tmp = tmp[tmp['start_time'] <= row['scheduled_gate_departure']]
            tmp = tmp[tmp['actual_end_time'] >= row['scheduled_gate_departure']]
            tmp = tmp.sort_index(by='start_time')
            i = 0
            for ix_tmp, row_tmp in tmp.iterrows():
                for j, val in enumerate(row_tmp):
                    departure_icing_delay["departure_icing_delay_{0}_{1}".format(i, tmp.columns[j])] = val
                i += 1
            if len(departure_icing_delay) > 0:
                departure_icing_delay['flight_history_id'] = ix
                departure_icing_delays.append(departure_icing_delay)
    arrival_ground_delays_df = pd.DataFrame(arrival_ground_delays)
    arrival_ground_delays_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    arrival_delays_df = pd.DataFrame(arrival_delays)
    arrival_delays_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    arrival_icing_delays_df = pd.DataFrame(arrival_icing_delays)
    arrival_icing_delays_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    departure_ground_delays_df = pd.DataFrame(departure_ground_delays)
    departure_ground_delays_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    departure_delays_df = pd.DataFrame(departure_delays)
    departure_delays_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    departure_icing_delays_df = pd.DataFrame(departure_icing_delays)
    departure_icing_delays_df.set_index('flight_history_id', inplace=True, verify_integrity=True)
    return (arrival_ground_delays_df, arrival_delays_df, arrival_icing_delays_df, departure_ground_delays_df, departure_delays_df, departure_icing_delays_df)

def process_data(df):
    diffs =  df['actual_runway_arrival'] - df['scheduled_runway_arrival']
    df['runway_arrival_diff'] = diffs
    diffs_gate = df['actual_gate_arrival'] - df['scheduled_gate_arrival']
    df['gate_arrival_diff'] = diffs_gate
    for column, series in df.iteritems():
        pass

def build_joined_data(subdirname):
    df = None
    date_prefix = subdirname
    try:
        df = pd.load("{0}_joined.p".format(subdirname))
    except Exception as e:
        df = None
    if df is None:
        print "Working on {0}".format(subdirname)
        df = get_flight_history()
        events = get_flight_history_events()
        asdi_disposition = get_asdi_disposition()
        asdi_merged = get_asdi_merged()
        joiners = [events, asdi_disposition, asdi_merged]
        per_flights = get_for_flights(df)
        joiners += per_flights
        df = df.join(joiners)
        metar_arrival = get_metar("arrival")
        metar_departure = get_metar("departure")
        df = pd.merge(df, metar_arrival, how="left", left_on="arrival_airport_icao_code", right_index=True)
        df = pd.merge(df, metar_departure, how="left", left_on="departure_airport_icao_code", right_index=True)
#        fbwind_arrival = get_fbwind("arrival")
#        fbwind_departure = get_fbwind("departure")
#        df = pd.merge(df, fbwind_arrival, how="left", left_on="arrival_airport_code", right_on="airport_code")
#        df = pd.merge(df, fbwind_departure, how="left", left_on="arrival_airport_code", right_on="airport_code")
        print df.columns
        pd.save(df, "{0}_joined.p".format(subdirname))
    return df

def build_joined_data_proxy(args):
    subdirname = args[0]
    return build_joined_data(subdirname)

if __name__ == '__main__':
    all_dfs = None
    pool_queue = []
    pool = Pool(processes=8)
    for subdirname in os.walk('{0}{1}'.format(data_prefix, data_rev_prefix)).next()[1]:
        pool_queue.append([subdirname])
    results = pool.map(build_joined_data_proxy, pool_queue, 1)
    for df in results:
        if all_dfs is None:
            all_dfs = df
        else:
            all_dfs = all_dfs.append(df)
    pd.save(all_dfs, "all_joined.p")
    all_dfs.to_csv("all_joined.csv")

