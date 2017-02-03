# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from datetime import datetime
try:
    from httplib import BadStatusLine
except ImportError:
    from http.client import BadStatusLine
import matplotlib.pyplot as plt
import numpy as np
import requests


class nwisError(Exception):
    def __init__(self, error_message):
        self.error_message = error_message

    def __str__(self):
        r""" This just returns one of the error messages listed in the checkresponse() function"""
        return repr(self.error_message)


class nwis(object):
    """Class to quickly download NWIS data using NWIS_ services
    .. _NWIS: https://waterservices.usgs.gov/

    :param service: name of web service to use; options are daily values ('dv'), instantaneous values ('iv'),
    site ('site'), and groundwater levels ('gwlevels')
    :param values: values for REST query; valid site is '01646500'; valid huc is '02070010'; valid bBox is
    '-83.000000,36.500000,-81.000000,38.500000'
    :param loc_type: filter type; valid values are 'huc', 'bBox', 'sites', and 'countyCd';
    see https://waterservices.usgs.gov/rest/IV-Service.html#MajorFilters for details
    :param **kwargs: other query parameters; optional

    """
    def __init__(self, service, values, loc_type, **kwargs):
        r""" Instantiates an instance of nwis"""
        self.service = service
        self.loc_type = loc_type
        self.values = self.parsesitelist(values)
        self.header = {'Accept-encoding': 'gzip'}
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

        :param response: The response from the API as a dictionary if the API code is 200.

        :returns: The response from the API as a dictionary if the API code is 200.

        .. raises:: nwisError; Gives different response messages depending on returned code from API.
        .. notes:: https://waterservices.usgs.gov/docs/portable_code.html
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

        :returns: response - A dictionary that has been dumped from JSON. '01585200'

        .. raises:: nwisError - Overrides the exceptions given in the requests library to give more custom error messages.
            Connection_error occurs if no internet connection exists. Timeout_error occurs if the request takes too
            long and redirect_error is shown if the url is formatted incorrectly.
        """
        http_error = 'Could not connect to the API. This could be because you have no internet connection, a parameter' \
                     ' was input incorrectly, or the API is currently down. Please try again.'

        kwargs[self.loc_type] = self.values
        kwargs['format'] = self.out_format

        if 'startDT' not in kwargs:
            kwargs['startDT'] = self.start_date
        if 'endDT' not in kwargs:
            kwargs['endDT'] = self.end_date

        total_url = self.url + self.service + '/?'
        response_ob = requests.get(total_url, params=kwargs, headers=self.header)
        if self.service != 'site':
            try:
                response_ob.json()
            except:
                raise nwisError("Could not decode response from {:} ".format(response_ob.url))

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

        :param values:
        :param type: list
        :returns: sitno (str); string with commas separating values

        :Example:
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

        :param kwargs: response of request
        :type kwargs: str

        .. returns:: df; Pandas DataFrame containing data downloaded from USGS
        """
        self.service = 'site'
        self.out_format = 'rdb'

        resp = self.get_response(**kwargs)
        print(resp.url)
        linefile = resp.iter_lines()
        numlist = []
        num = 0
        for line in linefile:
            if line.startswith(b"#"):
                numlist.append(num)
            num += 1
        numlist.append(numlist[-1] + 2)
        df = pd.read_table(resp.url, sep="\t", skiprows=numlist)
        return df


    def cleanGWL(self, df, colm='qualifiers',inplace=False):
        """Drops water level data of suspect quality based on lev_status_cd

        :param df: (pandas dataframe) groundwater dataframe
        :param colm: column to parse; defaults to 'qualifiers'

        :type colm: str
        :returns: sitno (str) - subset of input dataframe as new dataframe
        """
        if inplace:
            data = df
        else:
            data = df.copy(deep=True)
        data[colm] = data[colm].apply(lambda x: self.xcheck(x), 1)
        CleanData = data[~data[colm].isin(['Z', 'R', 'V', 'P', 'O', 'F', 'W', 'G', 'S', 'C', 'E', 'N'])]
        return CleanData


    def avg_wl(self, numObs=50, avgtype='stdWL', grptype='bytime', grper='12M'):
        """Calculates standardized statistics for a list of stations or a huc from the USGS

        :param numObs: minimum observations per site required to include site in analysis; default is 50
        :param avgtype: averaging technique for site data; options are 'avgDiffWL' and 'stdWL'; default is 'stWL'
        :param grptype: way to group the averaged data; options are 'bytime' or 'monthly' or user input; default 'bytime'
        :param grper: only used if 'bytime' called; defaults to '12M'; other times can be put in
        :return:
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

    def nwis_heat_map(self):
        from scipy.interpolate import griddata
        import matplotlib.cm as cm
        import matplotlib as mpl

        meth = 'linear'  # 'nearest'

        data = self.data

        if isinstance(data.index, pd.core.index.MultiIndex):
            data.index = data.index.droplevel(0)

        x = data.index.dayofyear
        y = data.index.year
        z = data.value.values

        xi = np.linspace(x.min(), x.max(), 1000)
        yi = np.linspace(y.min(), y.max(), 1000)
        zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method=meth)

        cmap = plt.cm.get_cmap('RdYlBu')
        norm = mpl.colors.Normalize(vmin=z.min(), vmax=z.max())
        #norm = mpl.colors.LogNorm(vmin=0.1, vmax=100000)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(z)

        br = plt.contourf(xi, yi, zi, color=m.to_rgba(z), cmap=cmap)
        # setup the colorbar


        cbar = plt.colorbar(m)
        cbar.set_label('Discharge (cfs)')

        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.yticks(range(y.min(), y.max()))

        mons = {'Apr': 90.25, 'Aug': 212.25, 'Dec': 334.25, 'Feb': 31, 'Jan': 1, 'Jul': 181.25, 'Jun': 151.25,
                'Mar': 59.25, 'May': 120.25,
                'Nov': 304.25, 'Oct': 273.25, 'Sep': 243.25}
        monnms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        plt.title(self.sites.station_nm[0].title())
        tickplc = []
        plt.xticks([mons[i] for i in monnms], monnms)
        plt.grid()

def get_elev(x, units='Meters'):
    """Uses USGS elevation service to retrieve elevation
    :param x: longitude and latitude of point where elevation is desired
    :type x: list
    :param units: units for returned value; defaults to Meters; options are 'Meters' or 'Feet'
    :type units: str

    :returns: ned float elevation of location in meters

    :Example:
        >>> get_elev([-111.21,41.4])
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
             print("Connection attempt {:} of 3 failed.".format(attempts))
             attempts += 1
             g = 0
    return g

def get_huc(x):
    """Receive the content of ``url``, parse it as JSON and return the object.

    :param x: [longitude, latitude]

    :returns: HUC12, HUC12_Name - 12 digit hydrologic unit code of location and the name associated with that code
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

    :param x: [longitude,latitude]
    :type x: str
    :returns: USGS-style site id (groundwater) DDMMSSDDDMMSS01
    """
    return dms(x[1]) + dms(x[0]) + '01'

def dms(dec):
    """converts decimal degree coordinates to a usgs station id
    :param dec: latitude or longitude value in decimal degrees
    :return: usgs id value

    .. note:: https://help.waterdata.usgs.gov/faq/sites/do-station-numbers-have-any-particular-meaning
    """
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


def get_recess(df, Q, freq='1D', inplace=False):
    """ Select the data when values are decreasing compared to previous time step

    :param df: DataFrame of hydro data
    :param Q: DataFrame field with discharge or water level data
    :param freq: Frequency of measurement of data; default is 1D
    :param inplace: If True, replace input DataFrame; default is false
    :return: DataFrame of all of the decreasing segments of the input DataFrame

    .. note:: from https://github.com/stijnvanhoey/hydropy
    """
    recess = df[Q].diff() < 0.0
    if inplace:
        df = df
    else:
        df = df[recess].copy()
    df = df.resample(freq).mean()
    return df


def RB_Flashiness(series):
    """Richards-Baker Flashiness Index for a series of daily mean discharges.
    https://github.com/hydrogeog/hydro/blob/master/hydro/core.py
    """
    Qsum = np.sum(series)           # sum of daily mean discharges
    Qpath = 0.0
    for i in range(len(series)):
        if i == 0:
            Qpath = series[i]       # first entry only
        else:
            Qpath += np.abs(series[i] - series[i-1])    # sum the absolute differences of the mean discharges
    return Qpath/Qsum


def flow_duration(series):
    """Creates the flow duration curve for a discharge dataset. Returns a pandas
    series whose index is the discharge values and series is exceedance probability.
    https://github.com/hydrogeog/hydro/blob/master/hydro/core.py
    """
    fd = pd.Series(series).value_counts()               # frequency of unique values
    fd.sort_index(inplace=True)                         # sort in order of increasing discharges
    fd = fd.cumsum()                                    # cumulative sum of frequencies
    fd = fd.apply(lambda x: 100 - x/fd.max() * 100)     # normalize
    return fd

def Lyne_Hollick(series, alpha=.925, direction='f'):
    """Recursive digital filter for baseflow separation. Based on Lyne and Hollick, 1979.
    series = array of discharge measurements
    alpha = filter parameter
    direction = (f)orward or (r)everse calculation
    https://github.com/hydrogeog/hydro/blob/master/hydro/core.py
    """
    series = np.array(series)
    f = np.zeros(len(series))
    if direction == 'f':
        for t in np.arange(1,len(series)):
            f[t] = alpha * f[t-1] + (1 + alpha)/2 * (series[t] - series[t-1])
            if series[t] - f[t] > series[t]:
                f[t] = 0
    elif direction == 'r':
        for t in np.arange(len(series)-2, 1, -1):
            f[t] = alpha * f[t+1] + (1 + alpha)/2 * (series[t] - series[t+1])
            if series[t] - f[t] > series[t]:
                f[t] = 0
    return np.array(series - f)

def Eckhardt(series, alpha=.98, BFI=.80):
    """Recursive digital filter for baseflow separation. Based on Eckhardt, 2004.
    series = array of discharge measurements
    alpha = filter parameter
    BFI = BFI_max (maximum baseflow index)
    https://github.com/hydrogeog/hydro/blob/master/hydro/core.py
    """
    series = np.array(series)
    f = np.zeros(len(series))
    f[0] = series[0]
    for t in np.arange(1,len(series)):
        f[t] = ((1 - BFI) * alpha * f[t-1] + (1 - alpha) * BFI * series[t]) / (1 - alpha * BFI)
        if f[t] > series[t]:
            f[t] = series[t]
    return f

def ratingCurve(discharge, stage):
    """Computes rating curve based on discharge measurements coupled with stage
    readings.
    discharge = array of measured discharges;
    stage = array of corresponding stage readings;
    Returns coefficients a, b for the rating curve in the form y = a * x**b
    https://github.com/hydrogeog/hydro/blob/master/hydro/core.py
    """
    from scipy.optimize import curve_fit

    exp_curve = lambda x, a, b: (a * x ** b)
    popt, pcov = curve_fit(exp_curve, stage, discharge)


    a = 0.0
    b = 0.0

    for i, j in zip(discharge, stage):
        a += (i - exp_curve(j, popt[0], popt[1]))**2
        b += (i - np.mean(discharge))**2
    r_squ = 1 - a / b


    return popt, r_squ
