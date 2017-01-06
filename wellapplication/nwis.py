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


class nwis(object):
    def __init__(self, service):
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
        self.header = {'Accept-encoding': 'gzip'}
        self.url = 'https://waterservices.usgs.gov/nwis/'
        self.geo_criteria = ['sites', 'stateCd', 'huc', 'countyCd', 'bBox']
        self.out_format = 'json'
        self.start_date = '1800-01-01'
        self.end_date = str(datetime.today().year) + '-' + str(datetime.today().month).zfill(2) + '-' + str(
            datetime.today().day).zfill(2)

    # ================================================================================================================ #
    # Functions:                                                                                                       #
    # ================================================================================================================ #

    @staticmethod
    def _checkresponse(response):
        r""" Returns the data requested by the other methods assuming the response from the API is ok. If not, provides
        error handling for all possible API errors. HTTP errors are handled in the get_response() function.

        Arguments:
        ----------
            None.

        Returns:
        --------
            The response from the API as a dictionary if the API code is 2.

        Raises:
        -------
            MesoPyError: Gives different response messages depending on returned code from API. If the response is 2,
            resultsError is displayed. For a response of 200, an authError message is shown. A ruleError is displayed
            if the code is 400, a formatError for -1, and catchError for any other invalid response.

        https://waterservices.usgs.gov/docs/portable_code.html
        """

        results_error = 'No results were found matching your query'
        auth_error = 'The token or API key is not valid, please contact Josh Clark at joshua.m.clark@utah.edu to ' \
                     'resolve this'
        rule_error = 'This request violates a rule of the API. Please check the guidelines for formatting a data ' \
                     'request and try again'
        catch_error = 'Something went wrong. Check all your calls and try again'

        if response.status_code == 200:
            print('connection successful')
            return response
        elif response.status_code == 403:
            print('the USGS has blocked your Internet Protocol (IP) address')
        elif response.status_code == 400:
            print('URL arguments are inconsistent')
        elif response.status_code == 404:
            print('the query expresses a combination of elements where data do not exist.')
        elif response.status_code == 500:
            print('there is a problem with the web service')
        elif response.status_code == 503:
            print('this application is down at the moment')
        else:
            raise MesoPyError(catch_error)

    def get_response(self, values, loc_type, **kwargs):
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
        kwargs[loc_type] = values

        kwargs['format'] = self.out_format

        if 'startDT' not in kwargs:
            kwargs['startDT'] = self.start_date
        if 'endDT' not in kwargs:
            kwargs['endDT'] = self.end_date

        total_url = self.url + self.service + '/?'
        response_ob = requests.get(total_url, params=kwargs, headers=self.header)

        # nwis_dict = response_ob.json()
        """
        # For python 2.7
        except AttributeError or NameError:
            try:
                qsp = urllib.urlencode(request_dict, doseq=True)
                resp = urllib2.urlopen(self.base_url + endpoint + '?' + qsp).read()
            except urllib2.URLError:
                raise MesoPyError(http_error)
        except urllib.error.URLError:
            raise MesoPyError(http_error)
        """
        return self._checkresponse(response_ob)  # nwis_dict#_checkresponse(json.loads(resp.decode('utf-8')))

    def _check_geo_param(self, arg_list):
        r""" Checks each function call to make sure that the user has provided at least one of the following geographic
        parameters: 'stid', 'state', 'country', 'county', 'radius', 'bbox', 'cwa', 'nwsfirezone', 'gacc', or 'subgacc'.

        Arguments:
        ----------
        arg_list: list, mandatory
            A list of kwargs from other functions.

        Returns:
        --------
            None.

        Raises:
        -------
            MesoPyError if no geographic search criteria is provided.

        """

        geo_func = lambda a, b: any(i in b for i in a)
        check = geo_func(self.geo_criteria, arg_list)
        if check is False:
            raise MesoPyError('No stations or geographic search criteria specified. Please provide one of the '
                              'following: stid, state, county, country, radius, bbox, cwa, nwsfirezone, gacc, subgacc')

    def get_nwis(self, values, loc_type, **kwargs):
        resp = self.get_response(values, loc_type, **kwargs)
        nwis_dict = resp.json()

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
            print('No Data!')
        return stations, data
