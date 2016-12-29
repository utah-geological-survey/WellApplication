# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa



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
    
def test_get_station_info():
    assert len(wa.get_station_info(['01585200','10136500'])) == 2
    
