# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa
import matplotlib
import matplotlib.pyplot as plt


def test_getelev():
    x = [-111.21,41.4]
    g = wa.getelev(x)
    assert g > 100.0

def test_gethuc():
    x = [-111.21,41.4]
    g = wa.get_huc(x)
    assert len(g[0])>0

def test_USGSID():
    x = [-111.21,41.4]
    g = wa.USGSID([-111.21,41.4])
    assert g == '412400111123601'

def test_get_nwis():
    val_list = '01585200'
    g = wa.get_nwis(val_list, 'dv_site', '2012-06-01', '2012-07-01')
    assert len(g) == 2

def test_get_nwis_huc():
    g = wa.get_nwis('16010204', 'gw_huc')
    assert len(g) == 2
    
def test_get_station_info():
    assert len(wa.get_station_info(['01585200','10136500'])) == 2
    
def test_cleanGWL():
    stat, levs = wa.get_nwis('16010204', 'gw_huc')
    levs2 = wa.cleanGWL(levs)
    assert type(levs2.qualifiers[0]) == str
    
def test_avg_wl():
    g = wa.avg_wl('16030006',numObs= 50, grptype = 'monthly', avgtype = 'avgDiffWL')
    assert len(g)==12

def test_mktest():
    x = range(0,100)
    trend = wa.MannKendall.mk_test(x,0.05)
    assert trend.trend == 'increasing'

def test_mkts():
    import pandas as pd
    # import data via rest query
    infile= r'http://nwis.waterdata.usgs.gov/usa/nwis/qwdata/?site_no=11530500&agency_cd=USGS&inventory_output=0&rdb_inventory_output=file&TZoutput=0&pm_cd_compare=Greater%20than&radio_parm_cds=parm_cd_list&radio_multiple_parm_cds=00665&qw_attributes=0&format=rdb&qw_sample_wide=wide&rdb_qw_attributes=0&date_format=YYYY-MM-DD&rdb_compression=value&submitted_form=brief_list'
    #response = urllib2.urlopen(infile)

    # designate header
    cols = ['datetime','agency','site','end_dt','end_tm','dtm','dtm_cd','coll_ent_cd','medium_cd','tu_id','body_part_id','PO4']

    # read data into a Pandas dataframe
    usgsP = pd.read_table(infile, skiprows=65, skipfooter=5, na_values=('-','','No Data','No data'),
                          parse_dates={'datetime':[2,3]}, engine='python')
    usgsP.columns = cols

    # set row names as datetime
    usgsP.set_index('datetime',inplace=True)

    # drop unused columns
    usgsP.drop(['agency','site','end_dt','end_tm','dtm','dtm_cd','coll_ent_cd','medium_cd','tu_id','body_part_id'],axis=1,inplace=True)

    # add year and month columns
    usgsP['month'] = usgsP.index.to_datetime().month.astype(int)
    usgsP['year'] = usgsP.index.to_datetime().year.astype(int)

    # filter data to relevant dates
    usgsP = usgsP[(usgsP['year']>=1972) & (usgsP.index.to_datetime() <= pd.datetime(1979,10,31))]

    # sort data by month then year for analysis
    usgsP.sort_values(by=['month','year'],axis=0, inplace=True)

    # remove strings from data column and convert to numbers
    usgsP['PO4'] = usgsP['PO4'].map(lambda x: x.strip('><E '))
    #usgsP['PO4'] = usgsP['PO4'].astype(float)

    usgsP['PO4'] = pd.to_numeric(usgsP['PO4'])
    usgsP.dropna(inplace=True)
    g = wa.MannKendall.mk_ts(usgsP, 'PO4', 'month', 'year',0.05)
    assert g.S == -87
