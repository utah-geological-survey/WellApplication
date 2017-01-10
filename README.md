[![DOI](https://zenodo.org/badge/48931715.svg)](https://zenodo.org/badge/latestdoi/48931715)
[![Build Status](https://travis-ci.org/inkenbrandt/WellApplication.svg?branch=master)](https://travis-ci.org/inkenbrandt/WellApplication)
[![PyPI version](https://badge.fury.io/py/WellApplication.svg)](https://badge.fury.io/py/WellApplication)
[![codecov](https://codecov.io/gh/inkenbrandt/WellApplication/branch/master/graph/badge.svg)](https://codecov.io/gh/inkenbrandt/WellApplication)

# Well Application

Set of tools for groundwater level and water chemistry analysis.  Allows for rapid download and graphing of data from the USGS NWIS database and the Water Quality Portal.


## Modules

### transport

This module:

* allows a user to upload data from an .xle file common with some water well transducers.

* matches well and barometric data to same sample intervals

* adjust with manual measurements

* removes skips and jumps from data

This class has functions used to import transducer data and condition it for analysis.

The most important function in this library is `new_xle_imp`, which uses the path and filename of an xle file, commonly produced by pressure transducers, to convert that file into a <a href=http://pandas.pydata.org/>Pandas</a> DataFrame.

A <a href=http://jupyter.org/> Jupyter Notebook</a> using some of the transport functions can be found <a href = http://nbviewer.jupyter.org/github/inkenbrandt/WellApplication/blob/master/docs/UMAR_WL_Data.ipynb>here</a>.

### usgs

This class has functions used to apply the USGS's rest-based api to download USGS data by leveraging  `urllib2 <https://docs.python.org/2/library/urllib2.html>`_
 package and <a href=http://pandas.pydata.org/>Pandas</a>.



A <a href=http://jupyter.org/> Jupyter (formerly IPython) Notebook</a> using some of the usgs functions can be found <a href=https://github.com/inkenbrandt/WellApplication/blob/master/docs/USGS_Interpolate.ipynb> here</a>.
