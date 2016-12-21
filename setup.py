import sys
from setuptools import setup, find_packages
# To use:
#	   python setup.py bdist --format=wininst

from wellapplication import __version__, __name__, __author__

# trap someone trying to install flopy with something other
#  than python 2 or 3
if not sys.version_info[0] in [2,3]:
    print('Sorry, wellapplication not supported in your Python version')
    print('  Supported versions: 2 and maybe 3')
    print('  Your version of Python: {}'.format(sys.version_info[0]))
    sys.exit(1)  # return non-zero value for failure

long_description = 'A tool for hydrogeologists to upload and display hydrographs and geochemical data'

try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except:
    pass

setup(name=__name__,
      description = 'Interface with xle files; analyze hydrographs; plot hydrographs; download USGS data',
      long_description = long_description,
      version = __version__,
      author = __author__,
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
                        "requests >= 2.11.1"],
      packages = find_packages(exclude=['contrib', 'docs', 'tests*']))





