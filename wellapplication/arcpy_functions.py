from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd


from .transport import *

try:
    import arcpy

    arcpy.env.overwriteOutput = True

except ImportError:
    pass


def imp_one_well(well_file, baro_file, man_startdate, man_start_level, man_endate, man_end_level,
                 conn_file_root,
                 wellid, be=None, well_table="UGGP.UGGPADMIN.UGS_NGWMN_Monitoring_Locations",
                 gw_reading_table="UGGP.UGGPADMIN.UGS_GW_reading", drift_tol=0.3, override=False):
    import arcpy
    arcpy.env.workspace = conn_file_root

    if os.path.splitext(well_file)[1] == '.xle':
        trans_type = 'Solinst'
    else:
        trans_type = 'Global Water'

    printmes('Trans type for well is {:}.'.format(trans_type))

    welltable = table_to_pandas_dataframe(well_table, query="AlternateID is not Null")

    well = new_trans_imp(well_file)
    baro = new_trans_imp(baro_file)


    corrwl = well_baro_merge(well, baro, vented=(trans_type != 'Solinst'))

    if be:
        corrwl = correct_be(wellid, welltable, corrwl, be=be)
        corrwl['corrwl'] = corrwl['BAROEFFICIENCYLEVEL']

    stickup, well_elev = get_stickup_elev(wellid, well_table)

    man = pd.DataFrame(
        {'DateTime': [man_startdate, man_endate], 'MeasuredDTW': [man_start_level, man_end_level]}).set_index(
        'DateTime')
    printmes(man)
    man['Meas_GW_Elev'] = well_elev - (man['MeasuredDTW'] - stickup)

    man['MeasuredDTW'] = man['MeasuredDTW'] * -1

    dft = fix_drift(corrwl, man, meas='corrwl', manmeas='MeasuredDTW')
    drift = round(float(dft[1]['drift'].values[0]), 3)
    printmes('Drift for well {:} is {:}.'.format(wellid, drift))
    df = dft[0]

    rowlist, fieldnames = prepare_fieldnames(df, wellid, stickup, well_elev)

    if drift <= drift_tol:
        edit_table(rowlist, gw_reading_table, fieldnames)
        printmes('Well {:} successfully imported!'.format(wellid))
    elif override == 1:
        edit_table(rowlist, gw_reading_table, fieldnames)
        printmes('Override initiated. Well {:} successfully imported!'.format(wellid))
    else:
        printmes('Well {:} drift greater than tolerance!'.format(wellid))
    return df, man, be, drift


def find_extreme(site_number, gw_table="UGGP.UGGPADMIN.UGS_GW_reading", extma='max'):
    """
    Find date extrema from a SDE table using query parameters
    :param site_number: LocationID of the site of interest
    :param gw_table: SDE table to be queried
    :param extma: options are 'max' (default) or 'min'
    :return: date of extrema, depth to water of extrema, water elevation of extrema
    """
    import arcpy
    from arcpy import env
    env.overwriteOutput = True

    if extma == 'max':
        sort = 'DESC'
    else:
        sort = 'ASC'
    query = "LOCATIONID = '{:}'".format(site_number)
    field_names = ['READINGDATE', 'LOCATIONID', 'DTWBELOWGROUNDSURFACE', 'WATERELEVATION']
    sql_sn = ('TOP 1', 'ORDER BY READINGDATE {:}'.format(sort))
    # use a search cursor to iterate rows
    dateval, dtw, wlelev = [], [], []

    envtable = os.path.join(env.workspace, gw_table)

    with arcpy.da.SearchCursor(envtable, field_names, query, sql_clause=sql_sn) as search_cursor:
        # iterate the rows
        for row in search_cursor:
            dateval.append(row[0])
            dtw.append(row[1])
            wlelev.append(row[2])
    if len(dateval) < 1:
        return None, 0, 0
    else:
        return dateval[0], dtw[0], wlelev[0]


def get_field_names(table):
    read_descr = arcpy.Describe(table)
    field_names = []
    for field in read_descr.fields:
        field_names.append(field.name)
    field_names.remove('OBJECTID')
    return field_names

def get_gap_data(site_number, enviro, gap_tol = 0.5,
                      gw_reading_table="UGGP.UGGPADMIN.UGS_GW_reading"):
    arcpy.env.workspace = enviro
    first_date = datetime.datetime(1900, 1, 1)
    last_date = datetime.datetime.now()

    query_txt = "LOCATIONID = '{:}' AND TAPE = 0"
    query = query_txt.format(site_number)

    sql_sn = (None, 'ORDER BY READINGDATE ASC')

    fieldnames = ['READINGDATE']

    #readings = wa.table_to_pandas_dataframe(gw_reading_table, fieldnames, query, sql_sn)

    dt = []

    # use a search cursor to iterate rows
    with arcpy.da.SearchCursor(gw_reading_table, 'READINGDATE', query, sql_clause=sql_sn) as search_cursor:
        # iterate the rows
        for row in search_cursor:
            # combine the field names and row items together, and append them
            dt.append(row[0])

    df = pd.Series(dt,name='DateTime')
    df = df.to_frame()
    df['hr_diff'] = df['DateTime'].diff()
    df.set_index('DateTime',inplace=True)
    df['julian'] = df.index.to_julian_date()
    df['diff'] = df['julian'].diff()
    df['is_gap'] =  df['diff'] > gap_tol
    def rowIndex(row):
        return row.name
    df['gap_end'] = df.apply(lambda x: rowIndex(x) if x['is_gap'] else pd.NaT, axis=1)
    df['gap_start'] = df.apply(lambda x: rowIndex(x) - x['hr_diff'] if x['is_gap'] else pd.NaT, axis=1)
    df = df[df['is_gap'] == True]
    return df



def table_to_pandas_dataframe(table, field_names=None, query=None, sql_sn=(None, None)):
    """
    Load data into a Pandas Data Frame for subsequent analysis.
    :param table: Table readable by ArcGIS.
    :param field_names: List of fields.
    :param query: SQL query to limit results
    :param sql_sn: sort fields for sql; see http://pro.arcgis.com/en/pro-app/arcpy/functions/searchcursor.htm
    :return: Pandas DataFrame object.
    """

    # if field names are not specified
    if not field_names:
        field_names = get_field_names(table)
    # create a pandas data frame
    df = pd.DataFrame(columns=field_names)

    # use a search cursor to iterate rows
    with arcpy.da.SearchCursor(table, field_names, query, sql_clause=sql_sn) as search_cursor:
        # iterate the rows
        for row in search_cursor:
            # combine the field names and row items together, and append them
            df = df.append(dict(zip(field_names, row)), ignore_index=True)

    # return the pandas data frame
    return df


def edit_table(df, gw_reading_table, fieldnames):
    """
    Edits SDE table by inserting new rows
    :param df: pandas DataFrame
    :param gw_reading_table: sde table to edit
    :param fieldnames: field names that are being appended in order of appearance in dataframe or list row
    :return:
    """

    table_names = get_field_names(gw_reading_table)

    for name in fieldnames:
        if name not in table_names:
            fieldnames.remove(name)
            printmes("{:} not in {:} fieldnames!".format(name, gw_reading_table))

    if len(fieldnames) > 0:
        subset = df[fieldnames]
        rowlist = subset.values.tolist()

        arcpy.env.overwriteOutput = True
        edit = arcpy.da.Editor(arcpy.env.workspace)
        edit.startEditing(False, False)
        edit.startOperation()

        cursor = arcpy.da.InsertCursor(gw_reading_table, fieldnames)
        for j in range(len(rowlist)):
            cursor.insertRow(rowlist[j])

        del cursor
        edit.stopOperation()
        edit.stopEditing(True)
    else:
        printmes('No data imported!')


def simp_imp_well(well_table, file, baro_out, wellid, manual, stbl_elev=True,
                  gw_reading_table="UGGP.UGGPADMIN.UGS_GW_reading", drift_tol=0.3, override=False):
    """
    Imports single well
    :param well_table: pandas dataframe of well data with ALternateID as index; needs altitude, be, stickup, and barolooger
    :param file: raw well file (xle, csv, or lev)
    :param baro_out: dictionary with barometer ID defining dataframe names
    :param wellid: unique ID of well field
    :param manual: manual data dataframe indexed by measure datetime
    :param stbl_elev:
    :param gw_reading_table:
    :param drift_tol:
    :param override:
    :return:
    """
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
        corrwl = well_baro_merge(well, baro_out[str(baroid)], barocolumn='MEASUREDLEVEL',
                                      vented=(trans_type != 'Solinst'))
    except:
        corrwl = well_baro_merge(well, baro_out['9003'], barocolumn='MEASUREDLEVEL',
                                      vented=(trans_type != 'Solinst'))

    # be, intercept, r = clarks(corrwl, 'barometer', 'corrwl')
    # correct barometric efficiency
    wls, be = correct_be(wellid, well_table, corrwl)

    # get manual groundwater elevations
    # man, stickup, well_elev = self.get_gw_elevs(wellid, well_table, manual, stable_elev = stbl_elev)
    stdata = well_table[well_table['WellID'] == str(wellid)]
    man_sub = manual[manual['LOCATIONID'] == int(wellid)]
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
    man_sub.loc[:, 'MeasuredDTW'] = man_sub['DTWBELOWCASING'] * -1
    man_sub.loc[:, 'Meas_GW_Elev'] = man_sub.loc[:, 'WATERELEVATION']
    #man_sub.loc[:, 'Meas_GW_Elev'] = man_sub['MeasuredDTW'].apply(lambda x: float(well_elev) + (x + float(stickup)),1)
    printmes('Stickup: {:}, Well Elev: {:}'.format(stickup, well_elev))

    # fix transducer drift

    dft = fix_drift(wls, man_sub, meas='BAROEFFICIENCYLEVEL', manmeas='MeasuredDTW')
    drift = np.round(float(dft[1]['drift'].values[0]), 3)

    df = dft[0]
    df.sort_index(inplace=True)
    first_index = df.first_valid_index()

    # Get last reading at the specified location
    read_max, dtw, wlelev = find_extreme(wellid)

    printmes("Last database date is {:}. First transducer reading is on {:}.".format(read_max, first_index))

    rowlist, fieldnames = prepare_fieldnames(df, wellid, stickup, well_elev)

    if (read_max is None or read_max < first_index) and (drift < drift_tol):
        edit_table(rowlist, gw_reading_table, fieldnames)
        printmes(arcpy.GetMessages())
        printmes("Well {:} imported.".format(wellid))
    elif override and (drift < drift_tol):
        edit_table(rowlist, gw_reading_table, fieldnames)
        printmes(arcpy.GetMessages())
        printmes("Override Activated. Well {:} imported.".format(wellid))
    elif drift > drift_tol:
        printmes('Drift for well {:} exceeds tolerance!'.format(wellid))
    else:
        printmes('Dates later than import data for well {:} already exist!'.format(wellid))
        pass

    # except (ValueError, ZeroDivisionError):

    #   drift = -9999
    #    df = corrwl
    #    pass
    return rowlist, man_sub, be, drift





def upload_bp_data(df, site_number, return_df=False, gw_reading_table="UGGP.UGGPADMIN.UGS_GW_reading"):
    import arcpy

    df.sort_index(inplace=True)
    first_index = df.first_valid_index()

    # Get last reading at the specified location
    read_max, dtw, wlelev = find_extreme(site_number)

    if read_max is None or read_max < first_index:

        df['MEASUREDLEVEL'] = df['Level']
        df['TAPE'] = 0
        df['LOCATIONID'] = site_number

        df.sort_index(inplace=True)

        fieldnames = ['READINGDATE', 'MEASUREDLEVEL', 'TEMP', 'LOCATIONID', 'TAPE']

        if 'Temperature' in df.columns:
            df.rename(columns={'Temperature': 'TEMP'}, inplace=True)

        if 'TEMP' in df.columns:
            df['TEMP'] = df['TEMP'].apply(lambda x: np.round(x, 4), 1)
        else:
            df['TEMP'] = None

        df.index.name = 'READINGDATE'

        subset = df.reset_index()

        edit_table(subset, gw_reading_table, fieldnames)

        if return_df:
            return df

    else:
        printmes('Dates later than import data for this station already exist!')
        pass


def get_location_data(site_number, enviro, first_date=None, last_date=None, limit=None,
                      gw_reading_table="UGGP.UGGPADMIN.UGS_GW_reading"):
    arcpy.env.workspace = enviro
    if not first_date:
        first_date = datetime.datetime(1900, 1, 1)
    elif type(first_date) == str:
        try:
            datetime.datetime.strptime(first_date, '%m/%d/%Y')
        except:
            first_date = datetime.datetime(1900, 1, 1)
    # Get last reading at the specified location
    if not last_date or last_date > datetime.datetime.now():
        last_date = datetime.datetime.now()

    query_txt = "LOCATIONID = '{:}' and (READINGDATE >= '{:%m/%d/%Y}' and READINGDATE <= '{:%m/%d/%Y}')"
    query = query_txt.format(site_number, first_date, last_date + datetime.timedelta(days=1))
    printmes(query)
    sql_sn = (limit, 'ORDER BY READINGDATE ASC')

    fieldnames = get_field_names(gw_reading_table)

    readings = table_to_pandas_dataframe(gw_reading_table, fieldnames, query, sql_sn)
    readings.set_index('READINGDATE', inplace=True)
    if len(readings) == 0:
        printmes('No Records for location {:}'.format(site_number))
    return readings
