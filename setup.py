from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
from setuptools import setup, find_packages

rootpath = os.path.abspath(os.path.dirname(__file__))
# To use:
#	   python setup.py bdist --format=wininst

# trap someone trying to install flopy with something other
#  than python 2 or 3
if not sys.version_info[0] in [2,3]:
    print('Sorry, wellapplication not supported in your Python version')
    print('  Supported versions: 2 and 3')
    print('  Your version of Python: {}'.format(sys.version_info[0]))
    sys.exit(1)  # return non-zero value for failure

version_file = open(os.path.join(rootpath, 'VERSION'))
version = version_file.read().strip()

long_description = 'A tool for hydrogeologists to upload and display hydrographs and geochemical data'

try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except:
    pass

setup(name='wellapplication',
      description = 'Interface with xle files; analyze hydrographs; plot hydrographs; download USGS data',
      long_description = long_description,
      version = version,
      author = 'Paul Inkenbrandt',
      author_email = 'paulinkenbrandt@utah.gov',
      url = 'https://github.com/inkenbrandt/WellApplication',
      license = 'LICENSE.txt',
      install_requires=["Pandas >= 0.18.0", 
                        "Numpy >= 1.10.0", 
                        "Matplotlib >= 1.4.3", 
                        "xmltodict >= 0.9.2",
                        "scipy >= 0.16.0",
                        "statsmodels >= 0.6.0",
                        "pyproj >= 1.9.4",
                        "requests >= 2.11.1",
                        "xlrd >= 0.9.4"],
      packages = find_packages(exclude=['contrib', 'docs', 'tests*']))





