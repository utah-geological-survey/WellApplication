# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa
import matplotlib
import pandas as pd

class test_usgs:
    
    def __init__(self):
        x = [-111.21,41.4]
        val_list = '01585200'
        val_huc = '16010204'
        stat, levs = wa.get_nwis(val_huc, 'gw_huc')
        g = wa.get_nwis(val_list, 'dv_site', '2012-06-01', '2012-07-01')
        f = wa.avg_wl('16030006',numObs= 50, grptype = 'monthly', avgtype = 'avgDiffWL')
        
    def test_getelev(self):
        g = wa.getelev(self, x)
        assert g > 100.0

    def test_gethuc(self, x):
        huc_data = wa.get_huc(x)
        assert len(g[0])>0

    def test_USGSID(self, x):
        usgs_id = wa.USGSID([-111.21,41.4])
        assert g == '412400111123601'

    def test_get_station_info(self):
        assert len(wa.get_station_info(['01585200','10136500'])) == 2
    
    def test_cleanGWL(self, levs):
        levs2 = wa.cleanGWL(levs)
        assert type(levs2.qualifiers[0]) == str

    def test_plt_avg_wl(self, f):
        assert type(wa.plt_avg_wl(f)) == matplotlib.figure.Figure

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
    
def test_pipe():
    pipr = wa.piper()
    Chem =  {'Type':[1,2,2,3], 'Cl':[1.72,0.90,4.09,1.52], 'HCO3':[4.02,1.28,4.29,3.04], 
             'SO4':[0.58,0.54,0.38,0.46], 'NaK':[1.40,0.90,3.38,2.86], 'Ca':[4.53,None,4.74,1.90], 
             'Mg':[0.79,0.74,0.72,0.66], 'EC':[672.0,308.0,884.0,542.0], 'NO3':[0.4,0.36,0.08,0.40], 
             'Sicc':[0.21,0.56,None,-0.41]}  
    chem = pd.DataFrame(Chem)
    pipr.piperplot(chem)
    assert type(pipr.plot) == matplotlib.figure.Figure
    
def test_fdc():
    sites,levs = wa.get_nwis('01585200')
    assert len(wa.fdc(levs,'value')[0]) > 100

def test_ganntplotter():
    sites,levs = wa.get_nwis('01585200')
    levs.drop(['qualifiers','site_no'],axis=1,inplace=True)
    levs['value'] = pd.to_numeric(levs['value'])
    gnt = wa.gantt(levs,stations=['value'])
    assert type(gnt.ganttPlotter()) == matplotlib.figure.Figure
    
def test_recess():
    cession = wa.graphs.recess()
    assert cession.ymd[0] > 2000

def test_reccur():
    cession = wa.graphs.recess()
    df = wa.get_nwis('01585200',selectType='dv_site', start_date='1968-01-01', end_date='1968-06-01')[1]
    type(cession.recession(df,'value',[1968,1,15],[1968,1,20])[1]) == pd.indexes.numeric.Float64Index
