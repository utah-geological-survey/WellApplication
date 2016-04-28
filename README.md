# Well Application


## Uses

This package:

* allows a user to upload data from an .xle file common with some water well transducers.

* matches well and barometric data to same sample intervals

* adjust with manual measurements

* removes skips and jumps from data

## Development

The most current iteration for this package is found at it's <a href=https://github.com/inkenbrandt/WellApplication>GitHub Repository</a>
.

## Modules

### transport

This class has functions used to import transducer data and condition it for analysis.

The most important function in this library is `new_xle_imp`, which uses the path and filename of an xle file, commonly produced by pressure transducers, to convert that file into a <a href=http://pandas.pydata.org/>Pandas</a> DataFrame.

A <a href=http://jupyter.org/> Jupyter (formerly IPython) Notebook</a> using some of the transport functions can be found <a href = http://nbviewer.jupyter.org/github/inkenbrandt/WellApplication/blob/master/docs/UMAR_WL_Data.ipynb>here</a>.

### usgs

This class has functions used to apply the USGS's rest-based api to download USGS data by leveraging  `urllib2 <https://docs.python.org/2/library/urllib2.html>`_
 package and <a href=http://pandas.pydata.org/>Pandas</a>.

A <a href=http://jupyter.org/> Jupyter (formerly IPython) Notebook</a> using some of the usgs functions can be found <a href=https://github.com/inkenbrandt/WellApplication/blob/master/docs/USGS_Interpolate.ipynb> here</a>.
