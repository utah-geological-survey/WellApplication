================
Well Application
================

.. image:: https://codecov.io/github/inkenbrandt/WellApplication/coverage.svg?branch=master
    :target: https://codecov.io/github/inkenbrandt/WellApplication?branch=master

.. image:: https://travis-ci.org/inkenbrandt/WellApplication.svg?branch=master
    :target: https://travis-ci.org/inkenbrandt/WellApplication

Uses
====

This package:

* allows a user to upload data from an .xle file common with some water well transducers.

* matches well and barometric data to same sample intervals

* adjust with manual measurements

* removes skips and jumps from data

Modules
=======

transport
---------

This class has functions used to import transducer data and condition it for analysis.

The most important function in this library is `new_xle_imp`, which uses the path and filename of an xle file, commonly produced by pressure transducers, to convert that file into a `Pandas <http://pandas.pydata.org/>`_ DataFrame.


usgsGis
-------

This class has functions used to apply the USGS's rest-based api to download USGS data by leveraging the urllib2 package.