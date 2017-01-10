# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
import pandas as pd
from datetime import datetime
from httplib import BadStatusLine
#import matplotlib.pyplot as plt
import numpy as np
import requests


class nwisError(Exception):
    def __init__(self, error_message):
        self.error_message = error_message

    def __str__(self):
        r""" This just returns one of the error messages listed in the checkresponse() function"""
        return repr(self.error_message)


class nwis(object):
    def __init__(self, service, values, loc_type, **kwargs):
        r""" Instantiates an instance of nwis

        Arguments:
        ----------
        token: string, mandatory
            Your API token that authenticates you for requests against MesoWest.mes

        Returns:
        --------
            None.

        Raises:
        -------
            None.
        """
        self.service = service
        self.loc_type = loc_type
        self.values = self.parsesitelist(values)
        #self.header = {'Accept-encoding': 'gzip'}
        self.url = 'https://waterservices.usgs.gov/nwis/'
        self.geo_criteria = ['sites', 'stateCd', 'huc', 'countyCd', 'bBox']
        self.out_format = 'json'
        self.start_date = '1800-01-01'
        self.input = kwargs
        self.end_date = str(datetime.today().year) + '-' + str(datetime.today().month).zfill(2) + '-' + str(
            datetime.today().day).zfill(2)
        self.sites, self.data = self.get_nwis(**kwargs)

    @staticmethod
    def _checkresponse(response):
        r""" Returns the data requested by the other methods assuming the response from the API is ok. If not, provides
        error handling for all possible API errors. HTTP errors are handled in the get_response() function.

        Arguments:
        ----------
            response: The response from the API as a dictionary if the API code is 200.

        Returns:
        --------
            The response from the API as a dictionary if the API code is 200.

        Raises:
        -------
            nwisError: Gives different response messages depending on returned code from API.

        https://waterservices.usgs.gov/docs/portable_code.html
        """

        if response.status_code == 200:
            print('connection successful')
            return response
        elif response.status_code == 403:
            raise nwisError('The USGS has blocked your Internet Protocol (IP) address')
        elif response.status_code == 400:
            raise nwisError('URL arguments are inconsistent')
        elif response.status_code == 404:
            raise nwisError('The query expresses a combination of elements where data do not exist.')
        elif response.status_code == 500:
            raise nwisError('There is a problem with the web service')
        elif response.status_code == 503:
            raise nwisError('This application is down at the moment')
        else:
            raise nwisError('Something went wrong.')

    def get_response(self, **kwargs):
        """ Returns a dictionary of data requested by each function.

        Arguments:
        ----------
        endpoint: string, mandatory
            Set in all other methods, this is the API endpoint specific to each function.
        request_dict: string, mandatory
            A dictionary of parameters that are formatted into the API call.

        Returns:
        --------
            response: A dictionary that has been dumped from JSON.
            '01585200'
        Raises:
        -------
            MesoPyError: Overrides the exceptions given in the requests library to give more custom error messages.
            Connection_error occurs if no internet connection exists. Timeout_error occurs if the request takes too
            long and redirect_error is shown if the url is formatted incorrectly.

        """
        http_error = 'Could not connect to the API. This could be because you have no internet connection, a parameter' \
                     ' was input incorrectly, or the API is currently down. Please try again.'
        # For python 3.4
        # try:
        kwargs[self.loc_type] = self.values
        kwargs['format'] = self.out_format

        if self.service == 'sites' and 'startDT' in kwargs and 'endDT' in kwargs:
            del kwargs['startDT']
            del kwargs['endDt']
        elif self.service == 'sites' and 'startDT' in kwargs:
            del kwargs['startDT']
        elif self.service == 'sites' and 'endDT' in kwargs:
            del kwargs['endDt']
        elif self.service == 'sites' and 'startDT' not in kwargs and 'endDT' not in kwargs:
            pass
        else:
            if 'startDT' not in kwargs:
                kwargs['startDT'] = self.start_date
            if 'endDT' not in kwargs:
                kwargs['endDT'] = self.end_date

        total_url = self.url + self.service + '/?'
        response_ob = requests.get(total_url, params=kwargs)# , headers=self.header)
        try:
            
            response_ob.json()
            
        except:
            raise nwisError('Could not decode response from {:}'.format(response_ob.url))
        return self._checkresponse(response_ob)

    def get_nwis(self, **kwargs):
        jsn_dict = self.get_response(**kwargs)
        nwis_dict = jsn_dict.json()
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
        elif len(dt) == 1:
            data = f[dt[0]['sourceInfo']['siteCode'][0]['value']]
            data['site_no'] = dt[0]['sourceInfo']['siteCode'][0]['value']
        else:
            data = None
            print('No Data!')
        return stations, data

    def parsesitelist(self, values):
        """Takes a list and turns it into a string format that can be used in the html REST format

        Args:
            values (list or array):
                list or array of ints or strings

        Returns:
            sitno (str):
                string with commas separating values

        Example::
            >>>parsesitelist([123,576,241])
            '123,576,241'
        """
        siteno = str(values).replace(" ", "")
        siteno = siteno.replace("]", "")
        siteno = siteno.replace("[", "")
        siteno = siteno.replace("','", ",")
        siteno = siteno.replace("'", "")
        siteno = siteno.replace('"', "")
        return siteno

    def get_info(self, **kwargs):
        """Downloads data from usgs service as text file; converted to Pandas DataFrame.
        Args:
            **kwargs (str):
                response of request

        Returns:
            df:
                Pandas DataFrame containing data downloaded from USGS
        """
        self.service = 'sites'
        self.out_format = 'rdb'

        resp = self.get_response(**kwargs)
        print(resp.url)
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


    def cleanGWL(self, df, colm='qualifiers'):
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
        data[colm] = data[colm].apply(lambda x: self.xcheck(x), 1)
        CleanData = data[~data[colm].isin(['Z', 'R', 'V', 'P', 'O', 'F', 'W', 'G', 'S', 'C', 'E', 'N'])]
        return CleanData


    def avg_wl(self, numObs=50, avgtype='stdWL', grptype='bytime', grper='12M'):
        """calculates standardized statistics for a list of stations or a huc from the USGS
        """

        data = self.cleanGWL(self.data)
        # stationWL = pd.merge(siteinfo, data, on = 'site_no')
        data.reset_index(inplace=True)
        data.set_index(['datetime'], inplace=True)
        # get averages by year, month, and site number
        site_size = data.groupby('site_no').size()
        wl_long = data[data['site_no'].isin(list(site_size[site_size >= numObs].index.values))]
        siteList = list(wl_long.site_no.unique())
        for site in siteList:
            mean = wl_long.ix[wl_long.site_no == site, 'value'].mean()
            std = wl_long.ix[wl_long.site_no == site, 'value'].std()
            wl_long.ix[wl_long.site_no == site, 'avgDiffWL'] = wl_long.ix[wl_long.site_no == site, 'value'] - mean
            wl_long.ix[wl_long.site_no == site, 'stdWL'] = wl_long.ix[wl_long.site_no == site, 'avgDiffWL'] / std

        if grptype == 'bytime':
            grp = pd.TimeGrouper(grper)
        elif grptype == 'monthly':
            grp = wl_long.index.month
        else:
            grp = grptype
        wl_stats = wl_long.groupby([grp])[avgtype].agg({'mean': np.mean, 'median': np.median,
                                                        'standard': np.std,
                                                        'cnt': (lambda x: np.count_nonzero(~np.isnan(x))),
                                                        'err_pls': (lambda x: np.mean(x) + (np.std(x) * 1.96)),
                                                        'err_min': (lambda x: np.mean(x) - (np.std(x) * 1.96))})

        return wl_stats

    def xcheck(self, x):
        """Converts empty list to empty string and filled list into string of first value"""
        if type(x) == list:
            if len(x) == 0:
                return ''
            else:
                return str(x[0])
        else:
            return x

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
        'f': 'pjson'}

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
    return dms(x[1]) + dms(x[0]) + '01'

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







