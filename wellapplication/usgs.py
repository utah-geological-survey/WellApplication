# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
import pandas as pd
from datetime import datetime
from httplib import BadStatusLine
import matplotlib.pyplot as plt
import numpy as np
import requests


def getelev(x, units='Meters'):
    """Uses USGS elevation service to retrieve elevation
    Args:
        x (array of floats):
            longitude and latitude of point where elevation is desired
        units (str):
            units for returned value; defaults to Meters; options are 'Meters' or 'Feet'
    Returns:
        ned float elevation of location in meters

    Example::
        >>> getelev([-111.21,41.4])
        1951.99
    """
    values = {
        'x': x[0],
        'y': x[1],
        'units': units,
        'output': 'json'
    }

    elev_url = 'http://ned.usgs.gov/epqs/pqs.php?'

    attempts = 0
    while attempts < 4:
        try:
            response = requests.get(elev_url, params=values).json()
            g = float(response['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
            break
        except(BadStatusLine):
            print "Connection attempt {:} of 3 failed.".format(attempts)
            attempts += 1
            g = 0
    return g


def get_huc(x):
    """Receive the content of ``url``, parse it as JSON and return the object.

    Args:
        x = [longitude, latitude]

    Returns:
        HUC12, HUC12_Name = 12 digit hydrologic unit code of location and the name associated with that code
    """
    values = {
        'geometry': '{:},{:}'.format(x[0], x[1]),
        'geometryType': 'esriGeometryPoint',
        'inSR': '4326',
        'spatialRel': 'esriSpatialRelIntersects',
        'returnGeometry': 'false',
        'outFields': 'HUC12,Name',
        'returnDistinctValues': 'true',
        'f': 'pjson'
    }

    huc_url = 'https://services.nationalmap.gov/arcgis/rest/services/USGSHydroNHDLarge/MapServer/10/query?'
    # huc_url2 = 'https://services.nationalmap.gov/arcgis/rest/services/nhd/mapserver/8/query?'
    response = requests.get(huc_url, params=values).json()
    return response['features'][0]['attributes']['HUC12'], response['features'][0]['attributes']['NAME']


def USGSID(x):
    """Parses decimal latitude and longitude values into DDMMSSDDDMMSS01 USGS site id.
    See https://help.waterdata.usgs.gov/faq/sites/do-station-numbers-have-any-particular-meaning for documentation.

    Args:
        x (list):
            [longitude,latitude]

    Returns:
        USGS-style site id (groundwater) DDMMSSDDDMMSS01
    """

    def dms(dec):
        DD = str(int(abs(dec)))
        MM = str(int((abs(dec) - int(DD)) * 60)).zfill(2)
        SS = str(int(round((((abs(dec) - int(DD)) * 60) - int(MM)) * 60, 0))).zfill(2)
        if SS == '60':
            MM = str(int(MM) + 1)
            SS = '00'
        if MM == '60':
            DD = str(int(DD) + 1)
            MM = '00'
        return DD + MM + SS

    return dms(x[1]) + dms(x[0]) + '01'


def get_nwis(val_list, selectType='dv_site', start_date='1800-01-01', end_date=''):
    """Request stream gauge data from the USGS NWIS.
    Args:
        val_list (str or int or list of either):
            can be a list or 8-digit hucs or stations or single
            a valid site is 01585200; a valid huc is 16010204
        selectType (str):
            options: 'dv_site','dv_huc','gw_site',or 'gw_huc'
        start_date (str):
           should take on the form yyyy-mm-dd; default is '1800-01-01'
        end_date (str):
            should take on the form yyyy-mm-dd; default is today
    Returns:
        a station and a data Pandas dataframe.
    Raises:
        ConnectionError  due to connection problems like refused connection
        or DNS Error.

    Example::
        >>> import wellapplication as wa
        >>> site, data = wa.get_nwis('01585200', 'dv_site', '2012-06-01', '2012-07-01')

    The specification for this service is located here:
    http://waterservices.usgs.gov/rest/IV-Service.html

    This function was adapted from: https://github.com/mroberge/hydrofunctions
    """
    val_list = parsesitelist(val_list)

    if end_date == '':
        dy = datetime.today()
        end_date = str(dy.year) + '-' + str(dy.month) + '-' + str(dy.day)

    header = {'Accept-encoding': 'gzip'}

    valdict = {
        'dv_site': {'format': 'json', 'sites': val_list, 'parameterCd': '00060',
                    'startDT': start_date, 'endDT': end_date},
        'dv_huc': {'format': 'json', 'huc': val_list, 'parameterCd': '00060',
                   'startDT': start_date, 'endDT': end_date},
        'gw_site': {'format': 'json', 'sites': val_list, 'siteType': 'GW', 'siteStatus': 'all',
                    'startDT': start_date, 'endDT': end_date},
        'gw_huc': {'format': 'json', 'huc': val_list, 'siteType': 'GW', 'siteStatus': 'all',
                   'startDT': start_date, 'endDT': end_date}
    }

    url = 'http://waterservices.usgs.gov/nwis/'

    if selectType == 'dv_site' or selectType == 'dv_huc':
        service = 'dv'
    elif selectType == 'gw_site' or selectType == 'gw_huc':
        service = 'gwlevels'
    url = url + service + '/?'
    response_ob = requests.get(url, params=valdict[selectType], headers=header)

    nwis_dict = response_ob.json()

    # dictionary from json object; each value in this dictionary is a station timeseries
    dt = nwis_dict['value']['timeSeries']

    station_id, lat, lon, srs, station_type, station_nm = [], [], [], [], [], []
    f = {}
    for i in range(len(dt)):
        station_id.append(dt[i]['sourceInfo']['siteCode'][0]['value'])
        lat.append(dt[i]['sourceInfo']['geoLocation'][u'geogLocation']['latitude'])
        lon.append(dt[i]['sourceInfo']['geoLocation'][u'geogLocation']['longitude'])
        srs.append(dt[i]['sourceInfo']['geoLocation'][u'geogLocation']['srs'])
        station_type.append(dt[i]['sourceInfo']['siteProperty'][0]['value'])
        station_nm.append(dt[i]['sourceInfo'][u'siteName'])

        df = pd.DataFrame(dt[i]['values'][0]['value'])
        df.index = pd.to_datetime(df.pop('dateTime'))
        df.value = df.value.astype(float)
        df.index.name = 'datetime'
        df.replace(to_replace='-999999', value=np.nan)
        f[dt[i]['sourceInfo']['siteCode'][0]['value']] = df
    stat_dict = {'site_no': station_id, 'dec_lat_va': lat, 'dec_long_va': lon, 'dec_coord_datum_cd': srs,
                 'station_nm': station_nm, 'data_type_cd': station_type}
    stations = pd.DataFrame(stat_dict)
    if len(dt) > 1:
        data = pd.concat(f)
        data.index.set_names('site_no', level=0, inplace=True)
    else:
        data = f[dt[0]['sourceInfo']['siteCode'][0]['value']]
        data['site_no'] = dt[0]['sourceInfo']['siteCode'][0]['value']
    return stations, data


def get_info(resp):
    """Downloads data from usgs service as text file; converted to Pandas DataFrame.
    Args:
        resp (str):
            response of request

    Returns:
        df:
            Pandas DataFrame containing data downloaded from USGS
    """
    linefile = resp.iter_lines()
    numlist = []
    num = 0
    for line in linefile:
        if line.startswith("#"):
            numlist.append(num)
        num += 1
    numlist.append(numlist[-1] + 2)
    df = pd.read_table(resp.url, sep="\t", skiprows=numlist)
    return df


def parsesitelist(ListOfSites):
    """Takes a list and turns it into a string format that can be used in the html REST format

    Args:
        ListOfSites (list or array):
            list or array of ints or strings

    Returns:
        sitno (str):
            string with commas separating values

    Example::
        >>>parsesitelist([123,576,241])
        '123,576,241'
    """
    siteno = str(ListOfSites).replace(" ", "")
    siteno = siteno.replace("]", "")
    siteno = siteno.replace("[", "")
    siteno = siteno.replace("','", ",")
    siteno = siteno.replace("'", "")
    siteno = siteno.replace('"', "")
    return siteno


def get_station_info(val_list, sitetype='sites', datatype=['all']):
    """Retrieve station info from a huc

    Arg:
        val_list (list):
            list of values to find sites
        sitetype(str):
            type of values to conduct query
            Options: sites, huc
        datatype (list or str):
            list of data types (default all); options include
            iv = instantaneous ,dv = daily values, sv = site visit, gw = groundwater level,
            qw = water quality, id =historical instantaneous

    Returns:
        Pandas DataFrame of sites
    """

    val_list = parsesitelist(val_list)
    sitetype = parsesitelist(sitetype)
    datatype = parsesitelist(datatype)

    valdict = {
        'sites': {'format': 'rdb,1.0', 'sites': val_list, 'hasDataTypeCd': datatype, 'siteOutput': 'expanded'},
        'huc': {'format': 'rdb,1.0', 'huc': val_list, 'hasDataTypeCd': datatype, 'siteOutput': 'expanded'},
    }

    url = "https://waterservices.usgs.gov/nwis/site/?"
    resp = requests.get(url, params=valdict[sitetype])
    siteinfo = get_info(resp)

    return siteinfo


def xcheck(x):
    """Converts empty list to empty string and filled list into string of first value"""
    if type(x) == list:
        if len(x) == 0:
            return ''
        else:
            return str(x[0])
    else:
        return x


def cleanGWL(df, colm='qualifiers'):
    """Drops water level data of suspect quality based on lev_status_cd
    returns Pandas DataFrame
    Args:
        df (pandas dataframe):
            groundwater dataframe
        colm (str):
            column to parse; defaults to 'qualifiers'
    Returns:
        sitno (str):
            subset of input dataframe as new dataframe

    """
    data = df.copy(deep=True)
    data[colm] = data[colm].apply(lambda x: xcheck(x), 1)
    CleanData = data[~data[colm].isin(['Z', 'R', 'V', 'P', 'O', 'F', 'W', 'G', 'S', 'C', 'E', 'N'])]
    return CleanData

def avg_wl(val_list, selectType='gw_huc',numObs = 50, avgtype = 'stdWL', grptype = 'bytime', grper = '12M'):
    """calculates standardized statistics for a list of stations or a huc from the USGS
    """
    siteinfo, data = get_nwis(val_list, selectType)
    data = cleanGWL(data)
    #stationWL = pd.merge(siteinfo, data, on = 'site_no')
    data.reset_index(inplace=True)
    data.set_index(['datetime'], inplace=True)
    # get averages by year, month, and site number
    site_size = data.groupby('site_no').size()
    wl_long = data[data['site_no'].isin(list(site_size[site_size >= numObs].index.values))]
    siteList = list(wl_long.site_no.unique())
    for site in siteList:
        mean = wl_long.ix[wl_long.site_no==site, 'value'].mean()
        std = wl_long.ix[wl_long.site_no==site, 'value'].std()
        wl_long.ix[wl_long.site_no==site, 'avgDiffWL'] = wl_long.ix[wl_long.site_no==site, 'value'] - mean
        wl_long.ix[wl_long.site_no==site, 'stdWL'] = wl_long.ix[wl_long.site_no==site, 'avgDiffWL']/std

    if grptype == 'bytime':
        grp = pd.TimeGrouper(grper)
    elif grptype == 'monthly':
        grp = wl_long.index.month
    else:
        grp = grptype
    wl_stats = wl_long.groupby([grp])[avgtype].agg({'mean': np.mean, 'median': np.median,
                                                                    'standard': np.std,
                                                                    'cnt': (lambda x: np.count_nonzero(~np.isnan(x))),
                                                                    'err_pls': (lambda x: np.mean(x)+(np.std(x)*1.96)),
                                                                    'err_min': (lambda x: np.mean(x)-(np.std(x)*1.96))})

    return wl_stats


def plt_avg_wl(grpd, beg='', end=''):
    x2 = grpd.index
    y2 = grpd['mean']
    y3 = grpd['median']
    snakegrp = grpd.median().to_frame()
    atitle = 'Deviation from mean water level (ft)'

    SIZE = 11
    matplotlib.rc('font', size=SIZE)
    matplotlib.rc('pdf', fonttype=42)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(x2, y3, '.-', color='green', label='Median')
    ax.plot(x2, y2, '.-', color='red', label='Average')
    ax.fill_between(x2, grpd['err_min'], grpd['err_pls'], alpha=0.2, label='2 Standard Deviations', linewidth=0)

    # ax1.set_ylim(3.5,-3.5)
    ax.set_ylabel(atitle, color='red')
    ax.invert_yaxis()
    ax.grid()

    if len(grpd) == 12 and type(grpd.index[0]) == np.int64:
        plt.xlim(0, 13)
        plt.xticks(range(1, 13, 1))
        plt.grid()
        ax.set_xlabel('Month')
        medlist = grpd['median'].values
        for i in range(len(grpd.index)):
            plt.text(grpd.index[i], medlist[i] - 3.2, 'n = ' + str(int(list(grpd['cnt'].values)[i])),
                     horizontalalignment='center')
        plt.legend()
    else:
        if beg == '':
            beg = grpd.index.min
        else:
            beg = pd.datetime(*beg)
        if end == '':
            end = pd.datetime.today()
        else:
            end = pd.datetime(*end)

        ax2 = ax.twinx()
        ax2.plot(x2, grpd['cnt'], '.-', label='Observations count')
        top = int(round(grpd['cnt'].max(), -1))
        ax2.set_ylim(0, top * 3)
        ax2.set_yticks(range(0, top + top / 10, top / 10))
        ax2.set_ylabel('Number of Observations', color='blue')
        ax2.yaxis.set_label_coords(1.04, 0.2)
        ax.set_xlim(beg, end)
        date_range = pd.date_range(beg, end, freq='36M')
        date_rang = date_range.map(lambda t: t.strftime('%Y-%m-%d'))
        ax.set_xticks(date_rang)
        ax.set_xlabel('date')
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.tight_layout()
    return fig
