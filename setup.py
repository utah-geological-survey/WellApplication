from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
from setuptools import setup, find_packages

if not sys.version_info[0] in [2,3]:
    print('Sorry, wellapplication not supported in your Python version')
    print('  Supported versions: 2 and 3')
    print('  Your version of Python: {}'.format(sys.version_info[0]))
    sys.exit(1)  # return non-zero value for failure

long_description = 'A tool for hydrogeologists to upload and display hydrographs and geochemical data'

try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except:
    pass

setup(name='wellapplication',
      description = 'Interface with xle files; analyze hydrographs; plot hydrographs; download USGS data',
      long_description = long_description,
      version = '0.5.2',
      author = 'Paul Inkenbrandt',
      author_email = 'paulinkenbrandt@utah.gov',
      url = 'https://github.com/inkenbrandt/WellApplication',
      license = 'LICENSE.txt',
      install_requires=["Pandas >= 0.16.0", 
                        "Numpy >= 0.7.0", 
                        "Matplotlib >= 1.1", 
                        "xmltodict >= 0.6.2",
                        "scipy >= 0.10.0",
                        "pyproj >= 1.9.4",
                        "requests >= 2.11.1",
                        "xlrd >= 0.5.4",
                        "statsmodels >= 0.6.0"],
      packages = find_packages(exclude=['contrib', 'docs', 'tests*']))





