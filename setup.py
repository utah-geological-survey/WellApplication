from setuptools import setup, find_packages

setup(name='WellApplication', 
      version='0.1.3', 
      author='Paul Inkenbrandt',
      author_email='paulinkenbrandt@utah.gov',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      url='https://github.com/inkenbrandt/WellApplication',
      license='LICENSE.txt',
      description='Interface with xle files; analyze hydrographs; plot hydrographs; download USGS data',
      install_requires=["Pandas >= 0.16.1", 
                        "Numpy >= 1.9.0", 
                        "Matplotlib >= 1.4.3", 
                        "xmltodict >= 0.9.2",
                        "scipy >= 0.13.3",
                        "pyproj >= 1.9.4"],
      classifiers = [
                    'Development Status :: 1 - Planning',
                    'Programming Language :: Python :: 2.7',
                    'Intended Audience :: Science/Research',
                    'Topic :: Scientific/Engineering :: GIS',
                    'Natural Language :: English',
                    'License :: OSI Approved :: MIT License',
                    'Programming Language :: Python :: 2.7'
                    ],
    keywords='hydrogeology hydrograph barocorrection fdc usgs',
    )
    
