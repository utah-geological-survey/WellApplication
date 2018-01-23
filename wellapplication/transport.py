from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as tick

from shutil import copyfile
import datetime
import os
import glob
import re
import xmltodict
from pylab import rcParams

rcParams['figure.figsize'] = 15, 10



def barodistance(wellinfo):
    """Determines Closest Barometer to Each Well using wellinfo DataFrame"""
    barometers = {'barom': ['pw03', 'pw10', 'pw19'], 'X': [240327.49, 271127.67, 305088.9],
                  'Y': [4314993.95, 4356071.98, 4389630.71], 'Z': [1623.079737, 1605.187759, 1412.673738]}
    barolocal = pd.DataFrame(barometers)
    barolocal = barolocal.reset_index()
    barolocal.set_index('barom', inplace=True)

    wellinfo['pw03'] = np.sqrt((barolocal.loc['pw03', 'X'] - wellinfo['UTMEasting']) ** 2 + \
                               (barolocal.loc['pw03', 'Y'] - wellinfo['UTMNorthing']) ** 2 + \
                               (barolocal.loc['pw03', 'Z'] - wellinfo['G_Elev_m']) ** 2)
    wellinfo['pw10'] = np.sqrt((barolocal.loc['pw10', 'X'] - wellinfo['UTMEasting']) ** 2 + \
                               (barolocal.loc['pw10', 'Y'] - wellinfo['UTMNorthing']) ** 2 + \
                               (barolocal.loc['pw10', 'Z'] - wellinfo['G_Elev_m']) ** 2)
    wellinfo['pw19'] = np.sqrt((barolocal.loc['pw19', 'X'] - wellinfo['UTMEasting']) ** 2 + \
                               (barolocal.loc['pw19', 'Y'] - wellinfo['UTMNorthing']) ** 2 + \
                               (barolocal.loc['pw19', 'Z'] - wellinfo['G_Elev_m']) ** 2)
    wellinfo['closest_baro'] = wellinfo[['pw03', 'pw10', 'pw19']].T.idxmin()
    return wellinfo

def rollmeandiff(df1, p1, df2, p2, win):
    """Returns the rolling mean difference of two columns from two different dataframes
    Args:
        df1 (object):
            dataframe 1
        p1 (str):
            column in df1
        df2 (object):
            dataframe 2
        p2 (str):
            column in df2
        win (int):
            window in days

    Return:
        diff (float):
            difference
    """
    win = win * 60 * 24
    df1 = df1.resample('1Min').mean()
    df1 = df1.interpolate(method='time')
    df2 = df2.resample('1Min').mean()
    df2 = df2.interpolate(method='time')
    df1['rm' + p1] = df1[p1].rolling(window=win, center=True).mean()
    df2['rm' + p2] = df2[p2].rolling(window=win, center=True).mean()
    df3 = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    df3 = df3[np.isfinite(df3['rm' + p1])]
    df4 = df3[np.isfinite(df3['rm' + p2])]
    df5 = df4['rm' + p1] - df4['rm' + p2]
    diff = round(df5.mean(), 3)
    del (df3, df4, df5)
    return diff

def jumpfix(df, meas, threashold=0.005, return_jump=False):
    """Removes jumps or jolts in time series data (where offset is lasting)
    Args:
        df (object):
            dataframe to manipulate
        meas (str):
            name of field with jolts
        threashold (float):
            size of jolt to search for
    Returns:
        df1: dataframe of corrected data
        jump: dataframe of jumps corrected in data
    """
    df1 = df.copy(deep=True)
    df1['delta' + meas] = df1.loc[:, meas].diff()
    jump = df1[abs(df1['delta' + meas]) > threashold]
    jump['cumul'] = jump.loc[:, 'delta' + meas].cumsum()
    df1['newVal'] = df1.loc[:, meas]

    for i in range(len(jump)):
        jt = jump.index[i]
        ja = jump['cumul'][i]
        df1.loc[jt:, 'newVal'] = df1[meas].apply(lambda x: x - ja, 1)
    df1[meas] = df1['newVal']
    if return_jump:
        print(jump)
        return df1, jump
    else:
        return df1

def smoother(df, p, win=30, sd=3):
    """Remove outliers from a pandas dataframe column and fill with interpolated values.
    warning: this will fill all NaN values in the DataFrame with the interpolate function

    Args:
        df (pandas.core.frame.DataFrame):
            Pandas DataFrame of interest
        p (string):
            column in dataframe with outliers
        win (int):
            size of window in days (default 30)
        sd (int):
            number of standard deviations allowed (default 3)

    Returns:
        Pandas DataFrame with outliers removed
    """
    df1 = df
    df1.loc[:, 'dp' + p] = df1.loc[:, p].diff()
    df1.loc[:, 'ma' + p] = df1.loc[:, 'dp' + p].rolling(window=win, center=True).mean()
    df1.loc[:, 'mst' + p] = df1.loc[:, 'dp' + p].rolling(window=win, center=True).std()
    for i in df.index:
        try:
            if abs(df1.loc[i, 'dp' + p] - df1.loc[i, 'ma' + p]) >= abs(df1.loc[i, 'mst' + p] * sd):
                df.loc[i, p] = np.nan
            else:
                df.loc[i, p] = df.loc[i, p]
        except ValueError:
            try:
                if abs(df1.loc[i, 'dp' + p] - df1.loc[i, 'ma' + p]) >= abs(df1.loc[:, 'dp' + p].std() * sd):
                    df.loc[i, p] = np.nan
                else:
                    df.loc[i, p] = df.loc[i, p]
            except ValueError:
                df.loc[i, p] = df.loc[i, p]

    try:
        df1 = df1.drop(['dp' + p, 'ma' + p, 'mst' + p], axis=1)
    except(NameError, ValueError):
        pass
    del df1
    try:
        df = df.drop(['dp' + p, 'ma' + p, 'mst' + p], axis=1)
    except(NameError, ValueError):
        pass
    df = df.interpolate(method='time', limit=30)
    df = df[1:-1]
    return df

def printmes(x):
    try:
        from arcpy import AddMessage
        AddMessage(x)
        print(x)
    except ModuleNotFoundError:
        print(x)

def compilefiles(searchdir,copydir,filecontains,filetypes=['lev','xle']):
    filecontains = list(filecontains)
    filetypes = list(filetypes)
    for pack in os.walk(searchdir):
        for name in filecontains:
            for i in glob.glob(pack[0]+'/'+'*{:}*'.format(name)):
                if i.split('.')[-1] in filetypes:
                    dater = str(datetime.datetime.fromtimestamp(os.path.getmtime(i)).strftime('%Y-%m-%d'))
                    rightfile = dater + "_" + os.path.basename(i)
                    if not os.path.exists(copydir):
                        print('Creating {:}'.format(copydir))
                        os.makedirs(copydir)
                    else:
                        pass
                    if os.path.isfile(os.path.join(copydir, rightfile)):
                        pass
                    else:
                        print(os.path.join(copydir, rightfile))
                        try:
                            copyfile(i, os.path.join(copydir, rightfile))
                        except:
                            pass
    printmes('Copy Complete!')
    return

def new_lev_imp(infile):
    with open(infile, "r") as fd:
        txt = fd.readlines()

    try:
        data_ind = txt.index('[Data]\n')
        # inst_info_ind = txt.index('[Instrument info from data header]\n')
        ch1_ind = txt.index('[CHANNEL 1 from data header]\n')
        ch2_ind = txt.index('[CHANNEL 2 from data header]\n')
        level = txt[ch1_ind + 1].split('=')[-1].strip().title()
        level_units = txt[ch1_ind + 2].split('=')[-1].strip().lower()
        temp = txt[ch2_ind + 1].split('=')[-1].strip().title()
        temp_units = txt[ch2_ind + 2].split('=')[-1].strip().lower()
        # serial_num = txt[inst_info_ind+1].split('=')[-1].strip().strip(".")
        # inst_num = txt[inst_info_ind+2].split('=')[-1].strip()
        # location = txt[inst_info_ind+3].split('=')[-1].strip()
        # start_time = txt[inst_info_ind+6].split('=')[-1].strip()
        # stop_time = txt[inst_info_ind+7].split('=')[-1].strip()

        df = pd.read_table(infile, parse_dates=[[0, 1]], sep='\s+', skiprows=data_ind + 2,
                           names=['Date', 'Time', level, temp],
                           skipfooter=1, engine='python')
        df.rename(columns={'Date_Time': 'DateTime'}, inplace=True)
        df.set_index('DateTime', inplace=True)

        if level_units == "feet" or level_units == "ft":
            df[level] = pd.to_numeric(df[level])
        elif level_units == "kpa":
            df[level] = pd.to_numeric(df[level]) * 0.33456
            printmes("Units in kpa, converting to ft...")
        elif level_units == "mbar":
            df[level] = pd.to_numeric(df[level]) * 0.0334552565551
        elif level_units == "psi":
            df[level] = pd.to_numeric(df[level]) * 2.306726
            printmes("Units in psi, converting to ft...")
        elif level_units == "m" or level_units == "meters":
            df[level] = pd.to_numeric(df[level]) * 3.28084
            printmes("Units in psi, converting to ft...")
        else:
            df[level] = pd.to_numeric(df[level])
            printmes("Unknown units, no conversion")

        if temp_units == 'Deg C' or temp_units == u'\N{DEGREE SIGN}' + u'C':
            df[temp] = df[temp]
        elif temp_units == 'Deg F' or temp_units == u'\N{DEGREE SIGN}' + u'F':
            printmes('Temp in F, converting to C')
            df[temp] = (df[temp] - 32.0) * 5.0 / 9.0
        df['name'] = infile
        return df
    except ValueError:
        printmes('File {:} has formatting issues'.format(infile))


def new_xle_imp(infile):
    """This function uses an exact file path to upload a xle transducer file.

    Args:
        infile (file):
            complete file path to input file

    Returns:
        A Pandas DataFrame containing the transducer data
    """
    # open text file
    with open(infile, "rb") as f:
        obj = xmltodict.parse(f, xml_attribs=True, encoding="ISO-8859-1")
    # navigate through xml to the data
    wellrawdata = obj['Body_xle']['Data']['Log']
    # convert xml data to pandas dataframe
    try:
        f = pd.DataFrame(wellrawdata)
    except ValueError:
        printmes('xle file {:} incomplete'.format(infile))
        return
    # CH 3 check
    try:
        ch3ID = obj['Body_xle']['Ch3_data_header']['Identification']
        f[str(ch3ID).title()] = f['ch3']
    except(KeyError, UnboundLocalError):
        pass

    # CH 2 manipulation
    try:
        ch2ID = obj['Body_xle']['Ch2_data_header']['Identification']
        f[str(ch2ID).title()] = f['ch2']
        ch2Unit = obj['Body_xle']['Ch2_data_header']['Unit']
        numCh2 = pd.to_numeric(f['ch2'])
        if ch2Unit == 'Deg C' or ch2Unit == u'\N{DEGREE SIGN}' + u'C':
            f[str(ch2ID).title()] = numCh2
        elif ch2Unit == 'Deg F' or ch2Unit == u'\N{DEGREE SIGN}' + u'F':
            printmes('Temp in F, converting to C')
            f[str(ch2ID).title()] = (numCh2 - 32) * 5 / 9
        f[str(ch2ID).title()] = pd.to_numeric(f[str(ch2ID).title()])
    except (KeyError,UnboundLocalError):
        printmes('No channel 2 for {:}'.format(infile))
    # CH 1 manipulation
    ch1ID = obj['Body_xle']['Ch1_data_header']['Identification']  # Usually level
    ch1Unit = obj['Body_xle']['Ch1_data_header']['Unit']  # Usually ft
    unit = str(ch1Unit).lower()

    if unit == "feet" or unit == "ft":
        f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])
    elif unit == "kpa":
        f[str(ch1ID).title()] = pd.to_numeric(f['ch1']) * 0.33456
        printmes("Units in kpa, converting to ft...")
    elif unit == "mbar":
        f[str(ch1ID).title()] = pd.to_numeric(f['ch1']) * 0.0334552565551
    elif unit == "psi":
        f[str(ch1ID).title()] = pd.to_numeric(f['ch1']) * 2.306726
        printmes("Units in psi, converting to ft...")
    elif unit == "m" or unit == "meters":
        f[str(ch1ID).title()] = pd.to_numeric(f['ch1']) * 3.28084
        printmes("Units in psi, converting to ft...")
    else:
        f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])
        printmes("Unknown units, no conversion")

    # add extension-free file name to dataframe
    f['name'] = infile.split('\\').pop().split('/').pop().rsplit('.', 1)[0]
    # combine Date and Time fields into one field
    f['DateTime'] = pd.to_datetime(f.apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
    f[str(ch1ID).title()] = pd.to_numeric(f[str(ch1ID).title()])


    try:
        ch3ID = obj['Body_xle']['Ch3_data_header']['Identification']
        f[str(ch3ID).title()] = pd.to_numeric(f[str(ch3ID).title()])
    except(KeyError, UnboundLocalError):
        pass

    f = f.reset_index()
    f = f.set_index('DateTime')
    f['Level'] = f[str(ch1ID).title()]
    f = f.drop(['Date', 'Time', '@id', 'ch1', 'ch2', 'index', 'ms'], axis=1)

    return f


def new_csv_imp(infile):
    """This function uses an exact file path to upload a csv transducer file.

    Args:
        infile (file):
            complete file path to input file

    Returns:
        A Pandas DataFrame containing the transducer data
    """
    f = pd.read_csv(infile, skiprows=1, parse_dates=[[0, 1]])
    # f = f.reset_index()
    f['DateTime'] = pd.to_datetime(f['Date_ Time'], errors='coerce')
    f = f[f.DateTime.notnull()]
    if ' Feet' in list(f.columns.values):
        f['Level'] = f[' Feet']
        f.drop([' Feet'], inplace=True, axis=1)
    elif 'Feet' in list(f.columns.values):
        f['Level'] = f['Feet']
        f.drop(['Feet'], inplace=True, axis=1)
    else:
        f['Level'] = f.iloc[:, 1]
    # Remove first and/or last measurements if the transducer was out of the water
    # f = dataendclean(f, 'Level')
    flist = f.columns.tolist()
    if ' Temp C' in flist:
        f['Temperature'] = f[' Temp C']
        f['Temp'] = f['Temperature']
        f.drop([' Temp C', 'Temperature'], inplace=True, axis=1)
    elif ' Temp F' in flist:
        f['Temperature'] = (f[' Temp F'] - 32) * 5 / 9
        f['Temp'] = f['Temperature']
        f.drop([' Temp F', 'Temperature'], inplace=True, axis=1)
    else:
        f['Temp'] = np.nan
    f.set_index(['DateTime'], inplace=True)
    f['date'] = f.index.to_julian_date().values
    f['datediff'] = f['date'].diff()
    f = f[f['datediff'] > 0]
    f = f[f['datediff'] < 1]
    # bse = int(pd.to_datetime(f.index).minute[0])
    # f = hourly_resample(f, bse)
    f.rename(columns={' Volts': 'Volts'}, inplace=True)
    f.drop([u'date', u'datediff', u'Date_ Time'], inplace=True, axis=1)
    return f


def new_trans_imp(infile):
    """This function uses an imports and cleans the ends of transducer file.

    Args:
        infile (file):
            complete file path to input file
        xle (bool):
            if true, then the file type should be xle; else it should be csv

    Returns:
        A Pandas DataFrame containing the transducer data
    """
    file_ext = os.path.splitext(infile)[1]
    try:
        if file_ext == '.xle':
            well = new_xle_imp(infile)
        elif file_ext == '.lev':
            well = new_lev_imp(infile)
        elif file_ext == '.csv':
            well = new_csv_imp(infile)
        else:
            printmes('filetype not recognized')
            pass
        return dataendclean(well, 'Level')
    except AttributeError:
        printmes('Bad File')
        return


def compilation(inputfile):
    """This function reads multiple xle transducer files in a directory and generates a compiled Pandas DataFrame.
    Args:
        inputfile (file):
            complete file path to input files; use * for wildcard in file name
    Returns:
        outfile (object):
            Pandas DataFrame of compiled data
    Example::
        >>> compilation('O:\\Snake Valley Water\\Transducer Data\\Raw_data_archive\\all\\LEV\\*baro*')
        picks any file containing 'baro'
    """

    # create empty dictionary to hold DataFrames
    f = {}

    # generate list of relevant files
    filelist = glob.glob(inputfile)

    # iterate through list of relevant files
    for infile in filelist:
        # run computations using lev files
        f[getfilename(infile)] = new_trans_imp(infile)
    # concatenate all of the DataFrames in dictionary f to one DataFrame: g
    g = pd.concat(f)
    # remove multiindex and replace with index=Datetime
    g = g.reset_index()
    g = g.set_index(['DateTime'])
    # drop old indexes
    g = g.drop(['level_0'], axis=1)
    # remove duplicates based on index then sort by index
    g['ind'] = g.index
    g.drop_duplicates(subset='ind', inplace=True)
    g.drop('ind', axis=1, inplace=True)
    g = g.sort_index()
    outfile = g
    return g


def dataendclean(df, x, inplace=False):
    """Trims off ends and beginnings of datasets that exceed 2.0 standard deviations of the first and last 30 values

    :param df: Pandas DataFrame
    :type df: pandas.core.frame.DataFrame
    :param x: Column name of data to be trimmed contained in df
    :type x: str
    :param inplace: if DataFrame should be duplicated
    :type inplace: bool

    :returns: df trimmed data
    :rtype: pandas.core.frame.DataFrame

    This function printmess a message if data are trimmed.
    """
    # Examine Mean Values
    if inplace:
        df = df
    else:
        df = df.copy()

    jump = df[abs(df.loc[:, x].diff()) > 1.0]
    try:
        for i in range(len(jump)):
            if jump.index[i] < df.index[50]:
                df = df[df.index > jump.index[i]]
                printmes("Dropped from beginning to " + str(jump.index[i]))
            if jump.index[i] > df.index[-50]:
                df = df[df.index < jump.index[i]]
                printmes("Dropped from end to " + str(jump.index[i]))
    except IndexError:
        printmes('No Jumps')
    return df


def getfilename(path):
    """This function extracts the file name without file path or extension

    Args:
        path (file):
            full path and file (including extension of file)

    Returns:
        name of file as string
    """
    return path.split('\\').pop().split('/').pop().rsplit('.', 1)[0]

def correct_be(site_number, well_table, welldata, be=None, meas='corrwl', baro='barometer'):
    if be:
        be = float(be)
    else:
        stdata = well_table[well_table['WellID'] == site_number]
        be = stdata['BaroEfficiency'].values[0]
    if be is None:
        be = 0
    else:
        be = float(be)

    if be == 0:
        welldata['BAROEFFICIENCYLEVEL'] = welldata[meas]
    else:
        welldata['BAROEFFICIENCYLEVEL'] = welldata[[meas, baro]].apply(lambda x: x[0] + be * x[1], 1)

    return welldata, be


def hourly_resample(df, bse=0, minutes=60):
    """
    resamples data to hourly on the hour
    Args:
        df:
            pandas dataframe containing time series needing resampling
        bse (int):
            base time to set; optional; default is zero (on the hour);
        minutes (int):
            sampling recurrence interval in minutes; optional; default is 60 (hourly samples)
    Returns:
        A Pandas DataFrame that has been resampled to every hour, at the minute defined by the base (bse)
    Description:
        see http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.resample.html for more info
        This function uses pandas powerful time-series manipulation to upsample to every minute, then downsample to every hour,
        on the hour.
        This function will need adjustment if you do not want it to return hourly samples, or iusgsGisf you are sampling more frequently than
        once per minute.
        see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    """

    df = df.resample('1Min').mean().interpolate(method='time', limit=90)

    df = df.resample(str(minutes) + 'Min', closed='left', label='left', base=bse).mean()
    return df


def well_baro_merge(wellfile, barofile, barocolumn='Level', wellcolumn='Level', outcolumn='corrwl',
                    vented=False, sampint=60):
    """Remove barometric pressure from nonvented transducers.
    Args:
        wellfile (pd.DataFrame):
            Pandas DataFrame of water level data labeled 'Level'; index must be datetime
        barofile (pd.DataFrame):
            Pandas DataFrame barometric data labeled 'Level'; index must be datetime
        sampint (int):
            sampling interval in minutes; default 60

    Returns:
        wellbaro (Pandas DataFrame):
           corrected water levels with bp removed
    """

    # resample data to make sample interval consistent
    baro = hourly_resample(barofile, 0, sampint)
    well = hourly_resample(wellfile, 0, sampint)

    # reassign `Level` to reduce ambiguity
    baro = baro.rename(columns={barocolumn: 'barometer'})

    if 'TEMP' in baro.columns:
        baro.drop('TEMP', axis=1, inplace=True)
    elif 'Temperature' in baro.columns:
        baro.drop('Temperature', axis=1, inplace=True)

    # combine baro and well data for easy calculations, graphing, and manipulation
    wellbaro = pd.merge(well, baro, left_index=True, right_index=True, how='inner')

    wellbaro['dbp'] = wellbaro['barometer'].diff()
    wellbaro['dwl'] = wellbaro[wellcolumn].diff()
    first_well = wellbaro[wellcolumn][0]

    if vented:
        wellbaro[outcolumn] = wellbaro[wellcolumn]
    else:
        wellbaro[outcolumn] = wellbaro[['dbp', 'dwl']].apply(lambda x: x[1] - x[0], 1).cumsum() + first_well
    wellbaro.loc[wellbaro.index[0], outcolumn] = first_well
    return wellbaro


def one_well(well_file, baro_file, well_table, man_startdate, man_start_level, man_endate,  man_end_level, wellid, be=None):
    """
    imports one well with the right files and manual data
    :param well_file: raw file of well data (lev, csv, or xle)
    :param baro_file: barometric pressure dictionary of pandas dataframes
    :param well_table: table of well information
    :param man_startdate: first manual measurement date near data interval
    :param man_start_level: first manual measurement near data interval
    :param man_endate: last manual measurement date near data interval
    :param man_end_level: last manual measurement near data interval
    :param wellid: unique id for well (alternate id in well table)
    :param be: barometric efficiency of well; default is None
    :return: rowlist, df, man, be, drift
    """

    if os.path.splitext(well_file)[1] == '.xle':
        trans_type = 'Solinst'
    else:
        trans_type = 'Global Water'

    printmes('Trans type for well is {:}.'.format(trans_type))


    well = new_trans_imp(well_file)
    baro = new_trans_imp(baro_file)


    corrwl = well_baro_merge(well, baro, vented=(trans_type != 'Solinst'))

    if be:
        corrwl = correct_be(wellid, corrwl, be=be)
        corrwl['corrwl'] = corrwl['BAROEFFICIENCYLEVEL']

    stickup, well_elev = get_stickup_elev(wellid, well_table)

    man = pd.DataFrame(
        {'DateTime': [man_startdate, man_endate],
         'MeasuredDTW': [man_start_level, man_end_level]}).set_index('DateTime')
    printmes(man)
    man['Meas_GW_Elev'] = well_elev - (man['MeasuredDTW'] - stickup)

    man['MeasuredDTW'] = man['MeasuredDTW'] * -1

    dft = fix_drift(corrwl, man, meas='corrwl', manmeas='MeasuredDTW')
    drift = round(float(dft[1]['drift'].values[0]), 3)
    printmes('Drift for well {:} is {:}.'.format(wellid, drift))
    df = dft[0]

    rowlist, fieldnames = prepare_fieldnames(df, wellid, stickup, well_elev)

    return rowlist, df, man, be, drift

def calc_dtw(df, stickup, well_elev, level='Level', dtw='DTW_WL'):
    """

    :param df: pandas DataFrame of processed well data
    :param stickup: raw transducer level from new_trans_imp, new_xle_imp, or new_csv_imp functions
    :param well_elev: well elevation
    :param level: raw transducer level from new_trans_imp, new_xle_imp, or new_csv_imp functions
    :param dtw: drift-corrected depth to water from fix_drift function
    :return:
    """
    df['MEASUREDLEVEL'] = df[level]
    df['MEASUREDDTW'] = df[dtw] * -1
    df['DTWBELOWCASING'] = df['MEASUREDDTW']
    df['DTWBELOWGROUNDSURFACE'] = df['MEASUREDDTW'].apply(lambda x: x - stickup, 1)
    df['WATERELEVATION'] = df['DTWBELOWGROUNDSURFACE'].apply(lambda x: well_elev - x, 1)
    return df

def prep_fields(df,wellid):
    """
    Prepares dataframe fields for import
    :param df: pandas dataframe containing groundwater levels
    :param wellid: unique well id
    :return: subset, fieldnames
    """
    df['TAPE'] = 0
    df['LOCATIONID'] = wellid

    df.sort_index(inplace=True)

    fieldnames = ['READINGDATE', 'MEASUREDLEVEL', 'MEASUREDDTW', 'DRIFTCORRECTION',
                  'TEMP', 'LOCATIONID', 'DTWBELOWCASING', 'BAROEFFICIENCYLEVEL',
                  'DTWBELOWGROUNDSURFACE', 'WATERELEVATION', 'TAPE']

    if 'Temperature' in df.columns:
        df.rename(columns={'Temperature': 'TEMP'}, inplace=True)

    if 'TEMP' in df.columns:
        df['TEMP'] = df['TEMP'].apply(lambda x: np.round(x, 4), 1)
    else:
        df['TEMP'] = None

    if 'BAROEFFICIENCYLEVEL' in df.columns:
        pass
    else:
        df['BAROEFFICIENCYLEVEL'] = 0
    # subset bp df and add relevant fields
    df.index.name = 'READINGDATE'

    subset = df.reset_index()

    return subset, fieldnames


def prepare_fieldnames(df, wellid, stickup, well_elev, level='Level', dtw='DTW_WL'):
    """
    This function adds the necessary field names to import well data into the SDE database.
    :param df: pandas DataFrame of processed well data
    :param wellid: wellid (alternateID) of well in Stations Table
    :param level: raw transducer level from new_trans_imp, new_xle_imp, or new_csv_imp functions
    :param dtw: drift-corrected depth to water from fix_drift function
    :return: processed df with necessary field names for import
    """

    df['MEASUREDLEVEL'] = df[level]
    df['MEASUREDDTW'] = df[dtw] * -1
    df['DTWBELOWCASING'] = df['MEASUREDDTW']
    df['DTWBELOWGROUNDSURFACE'] = df['MEASUREDDTW'].apply(lambda x: x - stickup, 1)
    df['WATERELEVATION'] = df['DTWBELOWGROUNDSURFACE'].apply(lambda x: well_elev - x, 1)
    df['TAPE'] = 0
    df['LOCATIONID'] = wellid

    df.sort_index(inplace=True)

    fieldnames = ['READINGDATE', 'MEASUREDLEVEL', 'MEASUREDDTW', 'DRIFTCORRECTION',
                  'TEMP', 'LOCATIONID', 'DTWBELOWCASING', 'BAROEFFICIENCYLEVEL',
                  'DTWBELOWGROUNDSURFACE', 'WATERELEVATION', 'TAPE']

    if 'Temperature' in df.columns:
        df.rename(columns={'Temperature': 'TEMP'}, inplace=True)

    if 'TEMP' in df.columns:
        df['TEMP'] = df['TEMP'].apply(lambda x: np.round(x, 4), 1)
    else:
        df['TEMP'] = None

    if 'BAROEFFICIENCYLEVEL' in df.columns:
        pass
    else:
        df['BAROEFFICIENCYLEVEL'] = 0
    # subset bp df and add relevant fields
    df.index.name = 'READINGDATE'

    subset = df.reset_index()

    return subset, fieldnames


def getwellid(infile, wellinfo):
    """Specialized function that uses a well info table and file name to lookup a well's id number"""
    m = re.search("\d", getfilename(infile))
    s = re.search("\s", getfilename(infile))
    if m.start() > 3:
        wellname = getfilename(infile)[0:m.start()].strip().lower()
    else:
        wellname = getfilename(infile)[0:s.start()].strip().lower()
    wellid = wellinfo[wellinfo['Well'] == wellname]['WellID'].values[0]
    return wellname, wellid

def get_stickup_elev(site_number, well_table):
    stdata = well_table[well_table['AltLocationID'] == str(site_number)]
    stickup = float(stdata['Offset'].values[0])
    well_elev = float(stdata['Altitude'].values[0])
    return stickup, well_elev


def get_gw_elevs(site_number, well_table, manual, stable_elev=True):
    """
    Gets basic well parameters and most recent groundwater level data for a well id for dtw calculations.
    :param site_number: well site number in the site table
    :param manual: Pandas Dataframe of manual data
    :param table: pandas dataframe of site table;
    :param lev_table: groundwater level table; defaults to "UGGP.UGGPADMIN.UGS_GW_reading"
    :return: stickup, well_elev, be, maxdate, dtw, wl_elev
    """

    stdata = well_table[well_table['WellID'] == str(site_number)]
    man_sub = manual[manual['Location ID'] == int(site_number)]
    well_elev = float(stdata['Altitude'].values[0])

    if stable_elev:
        stickup = float(stdata['Offset'].values[0])
    else:
        stickup = man_sub['Current Stickup Height']

    # manual = manual['MeasuredDTW'].to_frame()
    man_sub.loc[:, 'MeasuredDTW'] = man_sub['Water Level (ft)'] * -1
    man_sub.loc[:, 'Meas_GW_Elev'] = man_sub['MeasuredDTW'].apply(lambda x: well_elev + (x + stickup), 1)

    return man_sub, stickup, well_elev


def fix_well(well_table, file, baro_out, wellid, manual, stbl_elev=True, barocol='MEASUREDLEVEL', mancol = ''):

    # import well file
    well = new_trans_imp(file)

    file_ext = os.path.splitext(file)[1]
    if file_ext == '.xle':
        trans_type = 'Solinst'
    else:
        trans_type = 'Global Water'

    try:
        baroid = well_table.loc[wellid, 'BaroLoggerType']
        printmes('{:}'.format(baroid))
        corrwl = well_baro_merge(well, baro_out[str(baroid)], barocolumn=barocol,
                                      vented=(trans_type != 'Solinst'))
    except:
        corrwl = well_baro_merge(well, baro_out['9003'], barocolumn=barocol,
                                      vented=(trans_type != 'Solinst'))

    # be, intercept, r = clarks(corrwl, 'barometer', 'corrwl')
    # correct barometric efficiency
    wls, be = correct_be(wellid, well_table, corrwl)

    # get manual groundwater elevations
    # man, stickup, well_elev = self.get_gw_elevs(wellid, well_table, manual, stable_elev = stbl_elev)
    stdata = well_table[well_table['WellID'] == str(wellid)]
    man_sub = manual[manual['Location ID'] == int(wellid)]
    well_elev = float(stdata['Altitude'].values[0]) # Should be in feet

    if stbl_elev:
        if stdata['Offset'].values[0] is None:
            stickup = 0
            printmes('Well ID {:} missing stickup!'.format(wellid))
        else:
            stickup = float(stdata['Offset'].values[0])
    else:

        stickup = man_sub.loc[man_sub.last_valid_index(), 'Current Stickup Height']

    # manual = manual['MeasuredDTW'].to_frame()
    man_sub.loc[:, 'MeasuredDTW'] = man_sub['Water Level (ft)'] * -1
    man_sub.loc[:, 'Meas_GW_Elev'] = man_sub['MeasuredDTW'].apply(lambda x: float(well_elev) + (x + float(stickup)),
                                                                  1)
    printmes('Stickup: {:}, Well Elev: {:}'.format(stickup, well_elev))

    # fix transducer drift

    dft = fix_drift(wls, man_sub, meas='BAROEFFICIENCYLEVEL', manmeas='MeasuredDTW')
    drift = np.round(float(dft[1]['drift'].values[0]), 3)

    df = dft[0]
    df.sort_index(inplace=True)
    first_index = df.first_valid_index()

    rowlist, fieldnames = prepare_fieldnames(df, wellid, stickup, well_elev)

    #    pass
    return rowlist, man_sub, be, drift


def fcl(df, dtObj):
    """Finds closest date index in a dataframe to a date object
    Args:
        df:
            DataFrame
        dtObj:
            date object

    taken from: http://stackoverflow.com/questions/15115547/find-closest-row-of-dataframe-to-given-time-in-pandas
    """
    return df.iloc[np.argmin(np.abs(pd.to_datetime(df.index) - dtObj))]  # remove to_pydatetime()


def fix_drift(well, manualfile, meas='Level', corrwl='corrwl', manmeas='MeasuredDTW', outcolname='DTW_WL'):
    """Remove transducer drift from nonvented transducer data. Faster and should produce same output as fix_drift_stepwise
    Args:
        well (pd.DataFrame):
            Pandas DataFrame of merged water level and barometric data; index must be datetime
        manualfile (pandas.core.frame.DataFrame):
            Pandas DataFrame of manual measurements
        meas (str):
            name of column in well DataFrame containing transducer data to be corrected
        manmeas (str):
            name of column in manualfile Dataframe containing manual measurement data
        outcolname (str):
            name of column resulting from correction
    Returns:
        wellbarofixed (pandas.core.frame.DataFrame):
            corrected water levels with bp removed
        driftinfo (pandas.core.frame.DataFrame):
            dataframe of correction parameters
    """
    # breakpoints = self.get_breakpoints(wellbaro, manualfile)
    breakpoints = []
    manualfile.index = pd.to_datetime(manualfile.index)
    manualfile.sort_index(inplace=True)

    wellnona = well.dropna(subset=[corrwl])

    if manualfile.first_valid_index() > wellnona.first_valid_index():
        breakpoints.append(wellnona.first_valid_index())

    for i in range(len(manualfile)):
        breakpoints.append(fcl(wellnona, manualfile.index[i]).name)

    breakpoints = pd.Series(breakpoints)
    breakpoints = pd.to_datetime(breakpoints)
    breakpoints.sort_values(inplace=True)
    breakpoints.drop_duplicates(inplace=True)

    bracketedwls, drift_features = {}, {}

    if well.index.name:
        dtnm = well.index.name
    else:
        dtnm = 'DateTime'
        well.index.name = 'DateTime'

    manualfile.loc[:, 'julian'] = manualfile.index.to_julian_date()

    for i in range(len(breakpoints) - 1):
        # Break up pandas dataframe time series into pieces based on timing of manual measurements
        bracketedwls[i] = well.loc[
            (well.index.to_datetime() >= breakpoints[i]) & (well.index.to_datetime() <= breakpoints[i + 1])]
        df = bracketedwls[i]
        if len(df) > 0:
            df.sort_index(inplace=True)
            df.loc[:, 'julian'] = df.index.to_julian_date()

            last_trans = df.loc[df.last_valid_index(), meas]  # last transducer measurement
            first_trans = df.loc[df.first_valid_index(), meas]  # first transducer measurement

            first_man = fcl(manualfile, breakpoints[i])

            if df.first_valid_index() < manualfile.first_valid_index():
                first_man[manmeas] = None

            last_man = fcl(manualfile, breakpoints[i + 1])  # first manual measurment

            # intercept of line = value of first manual measurement
            if pd.isna(first_man[manmeas]):
                b = last_trans - last_man[manmeas]
                drift = 0.000001
            elif pd.isna(last_man[manmeas]):
                b = first_trans - first_man[manmeas]
                drift = 0.000001
            else:
                b = first_trans - first_man[manmeas]
                drift = ((last_trans - last_man[manmeas]) - b)

            # slope of line = change in difference between manual and transducer over time;
            m = drift / (df.loc[df.last_valid_index(), 'julian'] - df.loc[df.first_valid_index(), 'julian'])

            # datechange = amount of time between manual measurements
            df.loc[:, 'datechange'] = df['julian'].apply(lambda x: x - df.loc[df.index[0], 'julian'], 1)

            # bracketedwls[i].loc[:, 'wldiff'] = bracketedwls[i].loc[:, meas] - first_trans
            # apply linear drift to transducer data to fix drift; flipped x to match drift
            df.loc[:, 'DRIFTCORRECTION'] = df['datechange'].apply(lambda x: m * x, 1)
            df.loc[:, outcolname] = df[meas] - (df['DRIFTCORRECTION'] + b)

            drift_features[i] = {'t_beg': breakpoints[i], 'man_beg': first_man.name, 't_end': breakpoints[i + 1],
                                 'man_end': last_man.name,
                                 'intercept': b, 'slope': m,
                                 'first_meas': first_man[manmeas], 'last_meas': last_man[manmeas],
                                 'drift': drift, 'first_trans': first_trans, 'last_trans': last_trans}
        else:
            pass

    wellbarofixed = pd.concat(bracketedwls)
    wellbarofixed.reset_index(inplace=True)
    wellbarofixed.set_index(dtnm, inplace=True)
    drift_info = pd.DataFrame(drift_features).T

    return wellbarofixed, drift_info


def xle_head_table(folder):
    """Creates a Pandas DataFrame containing header information from all xle files in a folder
    Args:
        folder (directory):
            folder containing xle files
    Returns:
        A Pandas DataFrame containing the transducer header data
    Example::
        >>> xle_head_table('C:/folder_with_xles/')
    """
    # open text file
    df = {}
    for infile in os.listdir(folder):

        # get the extension of the input file
        filename, filetype = os.path.splitext(folder + infile)
        basename = os.path.basename(folder + infile)
        if filetype == '.xle':
            # open text file
            with open(folder + infile, "rb") as f:
                d = xmltodict.parse(f, xml_attribs=True, encoding="ISO-8859-1")
            # navigate through xml to the data
            data = list(d['Body_xle']['Instrument_info_data_header'].values()) + list(
                d['Body_xle']['Instrument_info'].values())
            cols = list(d['Body_xle']['Instrument_info_data_header'].keys()) + list(
                d['Body_xle']['Instrument_info'].keys())

            df[basename[:-4]] = pd.DataFrame(data=data, index=cols).T
    allwells = pd.concat(df)
    allwells.index = allwells.index.droplevel(1)
    allwells.index.name = 'filename'
    allwells['trans type'] = 'Solinst'
    allwells['fileroot'] = allwells.index
    allwells['full_filepath'] = allwells['fileroot'].apply(lambda x: folder + x + '.xle', 1)

    return allwells


def csv_info_table(folder):
    csv = {}
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    field_names = ['filename', 'Start_time', 'Stop_time']
    df = pd.DataFrame(columns=field_names)
    for file in files:
        fileparts = os.path.basename(file).split('.')
        filetype = fileparts[1]
        basename = fileparts[0]
        if filetype == 'csv':
            try:
                cfile = {}
                csv[basename] = new_csv_imp(os.path.join(folder, file))
                cfile['Battery_level'] = int(round(csv[basename].loc[csv[basename]. \
                                                   index[-1], 'Volts'] / csv[basename]. \
                                                   loc[csv[basename].index[0], 'Volts'] * 100, 0))
                cfile['Sample_rate'] = (csv[basename].index[1] - csv[basename].index[0]).seconds * 100
                cfile['filename'] = basename
                cfile['fileroot'] = basename
                cfile['full_filepath'] = os.path.join(folder, file)
                cfile['Start_time'] = csv[basename].first_valid_index()
                cfile['Stop_time'] = csv[basename].last_valid_index()
                cfile['Location'] = ' '.join(basename.split(' ')[:-1])
                cfile['trans type'] = 'Global Water'
                df = df.append(cfile, ignore_index=True)
            except:
                pass
    df.set_index('filename', inplace=True)
    return df, csv





class wellmod(object):
    def __init__(self):
        self.sde_conn = None
        self.well_file = None
        self.baro_file = None
        self.man_startdate = None
        self.man_enddate = None
        self.man_start_level = None
        self.man_end_level = None
        self.wellid = None
        self.xledir = None
        self.well_files = None
        self.wellname = None
        self.welldict = None
        self.filedict = None
        self.man_file = None
        self.save_location = None
        self.should_plot = None
        self.chart_out = None
        self.tol = None
        self.stbl = None
        self.ovrd = None
        self.toexcel = None
        self.baro_comp_file = None
        self.to_import = None
        self.idget = None

    def read_xle(self):
        well = new_xle_imp(self.well_file)
        well.to_csv(self.save_location)
        return


    def remove_bp(self):

        well = self.new_trans_imp(self.well_file)
        baro = self.new_trans_imp(self.baro_file)

        df = self.well_baro_merge(well, baro, barocolumn='Level', wellcolumn='Level', outcolumn='corrwl', vented=False,
                                  sampint=60)

        df.to_csv(self.save_location)

    def remove_bp_drift(self):

        well = self.new_trans_imp(self.well_file)
        baro = self.new_trans_imp(self.baro_file)

        man = pd.DataFrame(
            {'DateTime': [self.man_startdate, self.man_enddate],
             'MeasuredDTW': [self.man_start_level * -1, self.man_end_level * -1]}).set_index('DateTime')

        corrwl = self.well_baro_merge(well, baro, barocolumn='Level', wellcolumn='Level', outcolumn='corrwl',
                                      vented=False,
                                      sampint=60)

        dft = self.fix_drift(corrwl, man, meas='corrwl', manmeas='MeasuredDTW')
        drift = round(float(dft[1]['drift'].values[0]), 3)

        printmes("Drift is {:} feet".format(drift))
        dft[0].to_csv(self.save_location)

        if self.should_plot:
            pdf_pages = PdfPages(self.chart_out)

            # plot data
            df = dft[0]
            y1 = df['DTW_WL'].values
            y2 = df['barometer'].values
            x1 = df.index.values
            x2 = df.index.values

            x4 = man.index
            y4 = man['MeasuredDTW']
            fig, ax1 = plt.subplots()
            plt.xticks(rotation=70)
            ax1.scatter(x4, y4, color='purple')
            ax1.plot(x1, y1, color='blue', label='Water Level')
            ax1.set_ylabel('Depth to Water (ft)', color='blue')
            ax1.set_ylim(min(y1), max(y1))
            y_formatter = tick.ScalarFormatter(useOffset=False)
            ax1.yaxis.set_major_formatter(y_formatter)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Barometric Pressure (ft)', color='red')
            ax2.plot(x2, y2, color='red', label='Barometric pressure (ft)')
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc=3)
            plt.xlim(df.first_valid_index() - datetime.timedelta(days=3),
                     df.last_valid_index() + datetime.timedelta(days=3))

            pdf_pages.savefig(fig)
            plt.close()
            pdf_pages.close()

    def get_ftype(self, x):
        if x[1] == 'Solinst':
            ft = '.xle'
        else:
            ft = '.csv'
        return self.filedict.get(x[0] + ft)
