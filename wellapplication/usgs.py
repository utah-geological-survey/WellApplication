# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from datetime import datetime
from pylab import rcParams

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
        #self.url = 'https://waterservices.usgs.gov/nwis/'
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
        import dataretrieval.waterdata as waterdata
        import dataretrieval.nwis as legacy_nwis
        import pandas as pd
        import numpy as np

        print('Fetching data via hybrid engine (Legacy metadata + Modernized chunked data)...')
        
        # 1. Fetch the Site Metadata using the LEGACY engine
        loc_kwargs = {self.loc_type: self.values}
        
        try:
            print(f"Finding monitoring locations for {self.loc_type} = {self.values}...")
            sites_df, _ = legacy_nwis.get_info(**loc_kwargs)
        except Exception as e:
            raise nwisError(f"Could not fetch site metadata: {e}")

        if sites_df is None or sites_df.empty:
            print('No Data! (No sites found)')
            return None, None

        # Format legacy site IDs into modern "USGS-12345678" format
        site_ids = ["USGS-" + str(s) for s in sites_df['site_no'].dropna().unique()]

       # 2. Fetch measurements in CHUNKS to avoid 403 Server Limit errors
        meas_kwargs = {
            'parameter_code': '72019'  # <--- CRITICAL: This forces Depth to Water only
        }
        if 'startDT' in kwargs: meas_kwargs['start'] = kwargs['startDT']
        if 'endDT' in kwargs: meas_kwargs['end'] = kwargs['endDT']
        
        chunk_size = 50
        all_data_frames = []
        
        print(f"Fetching field measurements for {len(site_ids)} sites in chunks of {chunk_size}...")
        
        for i in range(0, len(site_ids), chunk_size):
            chunk = site_ids[i:i + chunk_size]
            try:
                # Fetch a single chunk
                chunk_df, _ = waterdata.get_field_measurements(
                    monitoring_location_id=chunk, 
                    **meas_kwargs
                )
                if chunk_df is not None and not chunk_df.empty:
                    all_data_frames.append(chunk_df)
            except Exception as e:
                # If a specific chunk fails, we report it but continue with others
                print(f"Warning: Failed to fetch chunk starting at index {i}: {e}")

        if not all_data_frames:
            print('No measurement data found for these sites!')
            return sites_df, None

        # Combine all successful chunks
        data_df = pd.concat(all_data_frames, ignore_index=True)
        print(f"Total data retrieved! Columns available: {list(data_df.columns)}")

        # 3. Dynamically find the Date column
        if 'phenomenon_time' in data_df.columns:
            data_df['datetime'] = pd.to_datetime(data_df['phenomenon_time'], errors='coerce')
        elif 'measurement_dt' in data_df.columns:
            data_df['datetime'] = pd.to_datetime(data_df['measurement_dt'], errors='coerce')
        else:
            possible_date_cols = [c for c in data_df.columns if 'date' in c.lower() or 'time' in c.lower() or c.endswith('_dt')]
            if possible_date_cols:
                data_df['datetime'] = pd.to_datetime(data_df[possible_date_cols[0]], errors='coerce')

        # 4. Dynamically find the Value and Quality columns
        if 'result_value' in data_df.columns:
            data_df.rename(columns={'result_value': 'value'}, inplace=True)
        elif 'lev_va' in data_df.columns:
            data_df.rename(columns={'lev_va': 'value'}, inplace=True)

        # Map the new OGC "result_qualifier" to "qualifiers" for cleanGWL compatibility
        if 'result_qualifier' in data_df.columns:
            data_df.rename(columns={'result_qualifier': 'qualifiers'}, inplace=True)
        elif 'lev_status_cd' in data_df.columns:
            data_df.rename(columns={'lev_status_cd': 'qualifiers'}, inplace=True)
        else:
            # If no qualifier column exists, create an empty string column 
            # so cleanGWL has a key to look at without crashing
            data_df['qualifiers'] = ""

        if 'value' in data_df.columns:
            data_df['value'] = pd.to_numeric(data_df['value'], errors='coerce')
            data_df['value'] = data_df['value'].where(data_df['value'] > -999, np.nan)

        # 5. Clean up site numbers and format metadata
        data_df['site_no'] = data_df.get('monitoring_location_id', data_df.get('site_no', pd.Series(dtype=str))).str.replace('USGS-', '')
        
        stat_dict = {
            'site_no': sites_df['site_no'],
            'dec_lat_va': sites_df['dec_lat_va'],
            'dec_long_va': sites_df['dec_long_va'],
            'dec_coord_datum_cd': sites_df['dec_coord_datum_cd'],
            'station_nm': sites_df['station_nm'],
            'data_type_cd': sites_df['site_tp_cd']
        }
        stations = pd.DataFrame(stat_dict)

        # 6. Final MultiIndex formatting
        unique_sites_found = list(data_df['site_no'].dropna().unique())
        if len(unique_sites_found) > 1:
            data = data_df.set_index(['site_no', 'datetime'])
        else:
            data = data_df.set_index('datetime')
            data['site_no'] = unique_sites_found[0] if unique_sites_found else None

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
        kwargs['siteOutput'] = 'expanded'
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

    @staticmethod
    def get_first_string(lst):
        """Function to get the first string from each list or return the string itself"""
        return lst[0] if isinstance(lst, list) and lst and all(isinstance(item, str) for item in lst) else lst

    def cleanGWL(self, df, colm='qualifiers', inplace=False):
        """Drops water level data of suspect quality based on qualifier codes"""
        if inplace:
            data = df
        else:
            data = df.copy(deep=True)
            
        # Use self. to reference the static method
        data[colm] = data[colm].apply(self.get_first_string)
        
        # Filter out the suspect codes
        CleanData = data[~data[colm].isin(['Z', 'R', 'V', 'P', 'O', 'F', 'W', 'G', 'S', 'C', 'E', 'N'])]
        return CleanData

    def my_agg(self, x):

        names = {
            'mean': x[self.avgtype].mean(numeric_only=True),
            'std': x[self.avgtype].std(numeric_only=True),
            'min': x[self.avgtype].min(numeric_only=True),
            'max': x[self.avgtype].max(numeric_only=True),
            'median': x[self.avgtype].median(numeric_only=True),
            'cnt': (np.count_nonzero(~np.isnan(x[self.avgtype]))),
            'err_pls': (np.mean(x[self.avgtype]) + (np.std(x[self.avgtype]) * 1.96)),
            'err_min': (np.mean(x[self.avgtype]) - (np.std(x[self.avgtype]) * 1.96))
            #'5 percent': np.percentile(x[self.avgtype], 5),
            #'95 percent': np.percentile(x[self.avgtype], 95)
        }

        return pd.Series(names, index=list(names.keys()))

    def avg_wl(self, numObs=50, avgtype='stdWL', grptype='bytime', grper='12ME'):
        self.avgtype = avgtype
        data = self.cleanGWL(self.data)
        
        data.reset_index(inplace=True)
        data.set_index(['datetime'], inplace=True)

        # 1. Filter out sites with too few observations
        site_counts = data.groupby('site_no')['value'].transform('count')
        wl_long = data[site_counts >= numObs].copy()

        if wl_long.empty:
            print("No sites met the minimum observation requirement.")
            return None

        # 2. Vectorized Statistics (No more loops!)
        # We group by site and transform to keep the original DataFrame shape
        grouped = wl_long.groupby('site_no')['value']
        
        wl_long['diff'] = grouped.diff()
        site_means = grouped.transform('mean')
        site_stds = grouped.transform('std')
        
        wl_long['avgDiffWL'] = wl_long['value'] - site_means
        wl_long['stdWL'] = wl_long['avgDiffWL'] / site_stds
        wl_long['cdm'] = wl_long.groupby('site_no')['avgDiffWL'].cumsum()
        
        # Standardized difference logic
        diff_grouped = wl_long.groupby('site_no')['diff']
        wl_long['avgDiff_dWL'] = wl_long['diff'] - diff_grouped.transform('mean')
        wl_long['std_dWL'] = wl_long['avgDiff_dWL'] / diff_grouped.transform('std')

        # 3. Grouping for output
        if grptype == 'bytime':
            grp = pd.Grouper(freq=grper)
        elif grptype == 'monthly':
            grp = wl_long.index.month
        else:
            grp = grptype

        # Reduce bias: average measurements per site per time step first
        site_time_avg = wl_long.groupby(['site_no', grp]).mean(numeric_only=True)
        
        # Calculate final stats across all sites for each time step
        wl_stats = site_time_avg.groupby(level=1).apply(self.my_agg)
        self.wl_stats = wl_stats

        return wl_stats

    def pltavgwl(self, maxdate = [0,0,0], mindate=[1950,1,1],):

        if maxdate[0] == 0:
            maxdate = [datetime.today().year,1,1]

        grpd = self.wl_stats
        x2 = grpd.index
        y3 = grpd['mean']
        y2 = grpd['median']

        fig = plt.figure()
        ax = fig.add_subplot(111)

        rcParams['figure.figsize'] = 15, 10
        rcParams['legend.numpoints'] = 1
        plt.plot(x2, y3, '+-', color='green', label='Median')
        ax.plot(x2, y2, '+-', color='red', label='Average')
        ax.fill_between(x2, grpd['err_min'], grpd['err_pls'], alpha=0.2, label='2 Standard Deviations', linewidth=0)

        ax.set_ylabel(self.avgtype, color='red')
        ax.invert_yaxis()
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(x2, grpd['cnt'], label='Number of Wells Observed')
        ax2.set_ylim(0, int(grpd['cnt'].max()) * 3)
        ax2.set_yticks(range(0, int(grpd['cnt'].max()), int(grpd['cnt'].max() / 10)))
        ax2.set_ylabel('Number of Wells Observed', color='blue')
        ax2.yaxis.set_label_coords(1.03, 0.2)
        ax.set_xlim(datetime(*mindate), datetime(*maxdate))
        date_range = pd.date_range('{:}-{:}-{:}'.format(*mindate), '{:}-{:}-{:}'.format(*maxdate), freq='36ME')
        date_range = date_range.map(lambda t: t.strftime('%Y-%m-%d'))
        ax.set_xticks(date_range)
        ax.set_xlabel('date')
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)

        return fig,ax,ax2

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
    """Uses the modernized USGS elevation service (v1)""" 
    
    # Map units to the code the API expects (0 for Feet, 1 for Meters)
    unit_code = '1' if units.lower() == 'meters' else '0'
    
    values = {
        'x': x[0],
        'y': x[1],
        'units': units, # The new API accepts 'Meters' or 'Feet' as strings too
        'output': 'json'
    }

    # Updated URL
    elev_url = 'https://epqs.nationalmap.gov/v1/json?'

    attempts = 0
    g = 0.0
    while attempts < 3:
        try:
            response = requests.get(elev_url, params=values, timeout=10).json()
            # The new JSON structure puts the value directly under 'value'
            g = float(response['value'])
            break
        except Exception as e:
            attempts += 1
            print(f"Elevation connection attempt {attempts} failed: {e}")
            
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
        'outFields': 'huc12,name',
        'returnDistinctValues': 'true',
        'f': 'pjson'}

    huc_url = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?'
    # huc_url = 'https://services.nationalmap.gov/arcgis/rest/services/USGSHydroNHDLarge/MapServer/10/query?'
    # huc_url2 = 'https://services.nationalmap.gov/arcgis/rest/services/nhd/mapserver/8/query?'
    response = requests.get(huc_url, params=values).json()
    return response['features'][0]['attributes']['huc12'], response['features'][0]['attributes']['name']

def get_fips(x):
    """Receive the content of ``url``, parse it as JSON and return the object.
    :param x: [longitude, latitude]
    :returns: tuple containing five digit county fips and county name
    """
    values = {
        'latitude': '{:}'.format(x[1]),
        'longitude': '{:}'.format(x[0]),
        'showall': 'true',
        'format': 'json'}

    huc_url = "http://data.fcc.gov/api/block/find?"
    response = requests.get(huc_url, params=values).json()
    return response['County']['FIPS'], response['County']['name']

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
