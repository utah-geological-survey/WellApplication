# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
import urllib2
import xmltodict
import pandas as pd
from datetime import datetime
from httplib import BadStatusLine
import matplotlib.pyplot as plt
import numpy as np
import avgMeths
import requests

def getelev(x):
    """Uses USGS elevation service to retrieve elevation
    Args:
        x (array of floats):
            longitude and latitude of point where elevation is desired

    Returns:
        ned float elevation of location in meters
    """
    elev = "http://ned.usgs.gov/epqs/pqs.php?x=" + str(x[0]) + "&y=" + str(x[1]) + "&units=Meters&output=xml"
    try:
        response = urllib2.urlopen(elev)
        html = response.read()
        d = xmltodict.parse(html)
        g = float(d['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
    except(BadStatusLine):
        try:
            response = urllib2.urlopen(elev)
            html = response.read()
            d = xmltodict.parse(html)
            g = float(d['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
        except(BadStatusLine):
            print "could not fetch {:}".format(html)
            g = 0
            pass
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
    #huc_url2 = 'https://services.nationalmap.gov/arcgis/rest/services/nhd/mapserver/8/query?'
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
            MM = str(int(MM)+1)
            SS = '00'
        if MM == '60':
            DD = str(int(DD)+1)
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

        df = pd.DataFrame(dt[i]['values'][0]['value'], columns=['dateTime', 'value'])
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

def getInfo(html):
    """Downloads data from usgs service as text file; converted to Pandas DataFrame.
    Args:
        html:
            location of data to be queried <http://waterservices.usgs.gov>

    Returns:
        df:
            Pandas DataFrame containing data downloaded from USGS
    """

    linefile = urllib2.urlopen(html).readlines()
    numlist = []
    num = 0
    for line in linefile:
        if line.startswith("#"):
            numlist.append(num)
        num += 1
    numlist.append(numlist[-1] + 2)
    df = pd.read_table(html, sep="\t", skiprows=numlist) 
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
        'sites': {'format': 'rdb,1.0', 'sites': val_list, 'hasDataTypeCd':  datatype, 'siteOutput':'expanded'},
        'huc': {'format': 'rdb,1.0', 'huc': val_list, 'hasDataTypeCd':  datatype, 'siteOutput':'expanded'},
    }

    url = "https://waterservices.usgs.gov/nwis/site/?"
    resp = requests.get(url, params=valdict[sitetype])
    linefile = resp.iter_lines()
    numlist = []
    num = 0
    for line in linefile:
        if line.startswith("#"):
            numlist.append(num)
        num += 1
    numlist.append(numlist[-1] + 2)
    siteinfo = pd.read_table(resp.url, sep="\t", skiprows=numlist) 

    return siteinfo


def cleanGWL(self, data):
    """
    Drops water level data of suspect quality based on lev_status_cd

    returns Pandas DataFrame
    """
    CleanData = data[~data['lev_status_cd'].isin(['Z', 'R', 'V', 'P', 'O', 'F', 'W', 'G', 'S', 'C', 'E', 'N'])]
    return CleanData

class usgs_stats:
    def WLStatdf(self, siteinfo, data):
        """Generates average water level statistics for a huc or list of hucs

        Args:
            siteinfo:
                Pandas DataFrame of site information of nwis sites (made using a get station info function)
            data:
                Pandas DataFrame of data from nwis sites (made using a get station data function)

        Returns:
            wlLongStatsGroups:
                Pandas DataFrame of standardized water levels over duration of measurement
            wlLongStatsGroups2:
                Pandas DataFrame of change in average water levels over duration of measurement
        """

        try:
            data.drop([u'agency_cd', u'site_tp_cd'], inplace=True, axis=1)
        except(ValueError):
            pass
        stationWL = pd.merge(data, siteinfo, on='site_no', how='left')

        stationWL['date'], stationWL['Year'], stationWL['Month'] = zip(
            *stationWL['lev_dt'].apply(lambda x: avgMeths.getyrmnth(x), 1))
        stationWL.reset_index(inplace=True)
        stationWL.set_index('date', inplace=True)
        stationWL = self.cleanGWL(stationWL)
        # get averages by year, month, and site number
        grpstat = stationWL.groupby('site_no')['lev_va'].agg(
            [np.std, np.mean, np.median, np.min, np.max, np.size]).reset_index()
        USGS_Site_Inf = stationWL.groupby('site_no')['lev_dt'].agg([np.min, np.max, np.size]).reset_index()
        USGS_Site_Info = USGS_Site_Inf[USGS_Site_Inf['size'] > 50]
        wlLong = stationWL[stationWL['site_no'].isin(list(USGS_Site_Info['site_no'].values))]
        wlLongStats = pd.merge(wlLong, grpstat, on='site_no', how='left')
        wlLongStats['stdWL'] = wlLongStats[['lev_va', 'mean', 'std']].apply(lambda x: avgMeths.stndrd(x), 1)
        wlLongStats['YRMO'] = wlLongStats[['Year', 'Month']].apply(lambda x: avgMeths.yrmo(x), 1)
        wlLongStats['date'] = wlLongStats[['Year', 'Month']].apply(lambda x: avgMeths.adddate(x), 1)
        self.wlMonthPlot = wlLongStats.groupby(['Month'])['stdWL'].mean().to_frame().plot()
        wlLongStats['levDiff'] = wlLongStats['lev_va'].diff()

        wlLongStatsGroups = wlLongStats.groupby(['date'])['stdWL'].agg({'mean': np.mean, 'median': np.median,
                                                                        'standard': np.std, 'cnt': (
                lambda x: np.count_nonzero(~np.isnan(x))),
                                                                        'err': (lambda x: 1.96 * avgMeths.sumstats(x))})
        wlLongStatsGroups2 = wlLongStats.groupby(['date'])['levDiff'].agg(
            {'mean': np.mean, 'median': np.median, 'standard': np.std,
             'cnt': (lambda x: np.count_nonzero(~np.isnan(x))), 'err': (lambda x: 1.96 * avgMeths.sumstats(x))})

        wlLongStatsGroups['meanpluserr'] = wlLongStatsGroups['mean'] + wlLongStatsGroups['err']
        wlLongStatsGroups['meanminuserr'] = wlLongStatsGroups['mean'] - wlLongStatsGroups['err']

        wlLongStatsGroups2['meanpluserr'] = wlLongStatsGroups2['mean'] + wlLongStatsGroups2['err']
        wlLongStatsGroups2['meanminuserr'] = wlLongStatsGroups2['mean'] - wlLongStatsGroups2['err']

        return wlLongStatsGroups, wlLongStatsGroups2

    def HUCplot(self, siteinfo, data):
        """Generates Statistics plots of NWIS WL data

        Args:
            siteinfo:
                pandas dataframe of site information of nwis sites (made using a get station info function)
            data:
                pandas dataframe of data from nwis sites (made using a get station data function)

        Returns:
            self.stand = standardized statistics
            self.diffs = difference statistics
            self.zPlot = seasonal variation plot
            self.wlPlot = plots of stadardized and differenced wl variations of duration of measurement
        """
        df1, df2 = self.WLStatdf(siteinfo, data)
        wlLongSt = df1[df1['cnt'] > 2]
        wlLongSt2 = df2[df2['cnt'] > 2]

        self.stand = wlLongSt
        self.diffs = wlLongSt2

        fig1 = plt.figure()
        x = wlLongSt.index
        y = wlLongSt['mean']
        plt.plot(x, y, label='Average Groundwater Level Variation')
        plt.fill_between(wlLongSt.index, wlLongSt['meanpluserr'], wlLongSt['meanminuserr'],
                         facecolor='blue', alpha=0.4, linewidth=0.5, label="Std Error")
        plt.grid(which='both')
        plt.ylabel('Depth to Water z-score')
        plt.xticks(rotation=45)
        self.zPlot = fig1

        fig2 = plt.figure()
        x = wlLongSt2.index
        y = wlLongSt2['mean']
        plt.plot(x, y, label='Average Groundwater Level Changes')
        plt.fill_between(wlLongSt.index, wlLongSt2['meanpluserr'], wlLongSt2['meanminuserr'],
                         facecolor='blue', alpha=0.4, linewidth=0.5, label="Std Error")
        plt.grid(which='both')
        plt.ylabel('Change in Average Depth to Water (ft)')
        plt.xticks(rotation=45)
        self.wlPlot = fig2

        return fig1, fig2, wlLongSt, wlLongSt2


