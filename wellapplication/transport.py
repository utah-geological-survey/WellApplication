    #-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
import re
import xmltodict
    
    
class transport:
    
    @staticmethod
    def jumpfix(df, meas, threashold=0.005):
        '''
        removes jumps or jolts in time series data (where offset is lasting)
        df = dataframe to manipulate
        meas = name of field with jolts
        threashold = size of jolt to search for
        '''
        df['delta'+meas] = df.loc[:,meas].diff()
        jump = df[abs(df['delta'+meas])>threashold]
        jump['cumul'] = jump.loc[:,'delta'+meas].cumsum()
        df['newVal'] = df.loc[:,meas]
        print jump
        for i in range(len(jump)):
            jt = jump.index[i]
            ja = jump['cumul'][i]
            df.loc[jt:,'newVal'] = df[meas].apply(lambda x: x-ja,1)
        df[meas]=df['newVal']
        return df
    
    @staticmethod
    def fcl(df, dtObj):
        '''
        finds closest date index in a dataframe to a date object
        
        df = dataframe
        dtObj = date object
        
        taken from: http://stackoverflow.com/questions/15115547/find-closest-row-of-dataframe-to-given-time-in-pandas
        '''
        return df.iloc[np.argmin(np.abs(pd.to_datetime(df.index) - dtObj))] # remove to_pydatetime()
    
    @staticmethod
    def getfilename(path):
        '''
        this function extracts the file name without file path or extension
		
		INPUT
		-----
		path = full path and file (including extension of file)
		
		RETURNS
		-------
		name of file as string
        '''        
        return path.split('\\').pop().split('/').pop().rsplit('.', 1)[0]
    
    @staticmethod
    def getwellid(infile, wellinfo):
        '''
        Specialized function that uses a well info table and file name to lookup a well's id number
        '''
        m = re.search("\d", transport.getfilename(infile))
        s = re.search("\s", transport.getfilename(infile))
        if m.start() > 3:
            wellname = transport.getfilename(infile)[0:m.start()].strip().lower()
        else:
            wellname = transport.getfilename(infile)[0:s.start()].strip().lower()
        wellid = wellinfo[wellinfo['Well']==wellname]['WellID'].values[0]
        return wellname, wellid
    
    @staticmethod    
    def hourly_resample(df,bse=0):
        '''
        INPUT
        -----
        df = Pandas DataFrame containing time series needing resampling
        bse = base time to set; default is zero (on the hour); 
        
        RETURNS
        -------
        df = A Pandas DataFrame that has been resampled to every hour, at the minute defined by the base (bse)
        
        DESCRIPTION
        -----------
        see http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.resample.html for more info
        
        This function uses pandas powerful time-series manipulation to upsample to every minute, then downsample to every hour, 
        on the hour.
        
        This function will need adjustment if you do not want it to return hourly samples, or if you are sampling more frequently than
        once per minute.
        
        see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        
        '''
        dfResamp = df.resample('1Min', closed='left', base=bse).first()
        dfResamp = dfResamp.interpolate(method='time')
        dfResamp = dfResamp.resample('60Min', closed='left', base=bse).first()
        return dfResamp
    
    @staticmethod 
    def hourly_resample_minutes(df,bse=0,minutes=60):
        '''
        INPUT
        -----
        df = pandas dataframe containing time series needing resampling
        bse = base time to set; optional; default is zero (on the hour); 
        minutes = sampling recurrence interval in minutes; optional; default is 60 (hourly samples)
        
        RETURNS
        -----
        A pandas dataframe that has been resampled to every hour, at the minute defined by the base (bse)
        
        DESCRIPTION
        -----
        see http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.resample.html for more info
        
        This function uses pandas powerful time-series manipulation to upsample to every minute, then downsample to every hour, 
        on the hour.
        
        This function will need adjustment if you do not want it to return hourly samples, or iusgsGisf you are sampling more frequently than
        once per minute.
        
        see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        
        '''
        df = df.resample('1Min') #you can make this smaller to accomodate for a higher sampling frequency
        df = df.interpolate(method='time', limit=90) #http://pandas.pydata.org/pandas-docs/dev/generated/pandas.Series.interpolate.html
        df = df.resample(str(minutes)+'Min', closed='left',label='left', base=bse).first() #modify '60Min' to change the resulting frequency
        return df
    
    @staticmethod 
    def dataendclean(df,x):
        '''
        trims off ends and beginnings of datasets that exceed 2.0 standard deviations of the first and last 30 values
		
		INPUT
		-----
		df = Pandas DataFrame
		x = Column name of data to be trimmed contained in df
        
		RETURNS
		-------
		df = trimmed data
		
		This function prints a message if data are trimmed.
		'''
        ## Examine Mean Values
        jump = df[abs(df.loc[:,x].diff()) > 1.0]
        try:
            for i in range(len(jump)):
                if jump.index[i] < df.index[50]:
                    df = df[df.index > jump.index[i]]
                    print("Dropped from beginning to " + str(jump.index[i]))
                if jump.index[i] > df.index[-50]:
                    df = df[df.index < jump.index[i]]
                    print("Dropped from end to " + str(jump.index[i]))
        except(IndexError):
            print('No Jumps')
        return df
    

    @staticmethod 
    def new_xle_imp(infile):
        '''
        This function uses an exact file path to upload a xle transducer file. 
        
		INPUT
		-----
        infile = complete file path to input file
        
        RETURNS
		-------
        A Pandas DataFrame containing the transducer data
        '''
        # open text file
        with open(infile) as fd:
            # parse xml to a dictionary, encode for degree signs
            obj = xmltodict.parse(fd.read(), encoding="ISO-8859-1")
        # navigate through xml to the data
        wellrawdata = obj['Body_xle']['Data']['Log']
        # convert xml data to pandas dataframe
        f = pd.DataFrame(wellrawdata)
             
        #CH 3 check
        try:
            ch3ID = obj['Body_xle']['Ch3_data_header']['Identification']
            f[str(ch3ID).title()] = f['ch3']
        except(KeyError,UnboundLocalError):
            pass
        
        #CH 2 manipulation
        ch2ID = obj['Body_xle']['Ch2_data_header']['Identification']     
        f[str(ch2ID).title()] = f['ch2']
        ch2Unit = obj['Body_xle']['Ch2_data_header']['Unit']
        numCh2 = pd.to_numeric(f['ch2'])
        if ch2Unit == 'Deg C' or ch2Unit == u'\N{DEGREE SIGN}' + u'C':
            f[str(ch2ID).title()] = numCh2
        elif ch2Unit == 'Deg F' or ch2Unit == u'\N{DEGREE SIGN}' + u'F': 
            print('Temp in F, converting to C')
            f[str(ch2ID).title()] = (numCh2-32)*5/9
    
        #CH 1 manipulation    
        ch1ID =  obj['Body_xle']['Ch1_data_header']['Identification'] # Usually level
        ch1Unit = obj['Body_xle']['Ch1_data_header']['Unit'] # Usually ft
        unit = str(ch1Unit).lower()
    
        if unit == "feet" or unit == "ft":
            f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])
        elif unit == "kpa":
            f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])*0.33456
            print("Units in kpa, converting to ft...")
        elif unit == "mbar":
            f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])*0.0334552565551
        elif unit == "psi":
            f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])*2.306726
            print("Units in psi, converting to ft...")
        elif unit == "m" or unit == "meters":
            f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])*3.28084
            print("Units in psi, converting to ft...")
        else:
            f[str(ch1ID).title()] = pd.to_numeric(f['ch1'])
            print("Unknown units, no conversion")
            
        # add extension-free file name to dataframe
        f['name'] = transport.getfilename(infile)
        # combine Date and Time fields into one field
        f['DateTime'] = pd.to_datetime(f.apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
        f[str(ch1ID).title()] =  pd.to_numeric(f[str(ch1ID).title()])
        f[str(ch2ID).title()] =  pd.to_numeric(f[str(ch2ID).title()])
        
        try:
            ch3ID = obj['Body_xle']['Ch3_data_header']['Identification']
            f[str(ch3ID).title()] =  pd.to_numeric(f[str(ch3ID).title()])
        except(KeyError,UnboundLocalError):
            pass
    
        f = f.reset_index()
        f = f.set_index('DateTime')
        f['Level'] = f[str(ch1ID).title()]
        f['MeasuredLevel'] = f['Level']
        f = f.drop(['Date','Time','@id','ch1','ch2','index','ms'],axis=1)
        
         
        return f
    
    @staticmethod 
    def compilation(inputfile):
        """
        This function reads multiple xle transducer files in a directory and generates a compiled Pandas DataFrame.
        
        inputfile = complete file path to input files; use * for wildcard in file name
            example -> 'O:\\Snake Valley Water\\Transducer Data\\Raw_data_archive\\all\\LEV\\*baro*' picks any file containing 'baro'
        
        """
            
        # create empty dictionary to hold DataFrames
        f={}
    
        # generate list of relevant files
        filelist = glob.glob(inputfile)
    
        # iterate through list of relevant files
        for infile in filelist:
            # get the extension of the input file
            filetype = os.path.splitext(infile)[1]
            # run computations using lev files
            if filetype=='.lev':
                # open text file
                with open(infile) as fd:
                    # find beginning of data
                    indices = fd.readlines().index('[Data]\n')
    
                # convert data to pandas dataframe starting at the indexed data line
                f[transport.getfilename(infile)] = pd.read_table(infile, parse_dates=True, sep='     ', index_col=0,
                                               skiprows=indices+2, names=['DateTime','Level','Temperature'], skipfooter=1,engine='python')
                # add extension-free file name to dataframe
                f[transport.getfilename(infile)]['name'] = transport.getfilename(infile)
                f['Level'] = pd.to_numeric(f['Level'])
                f['Temperature'] = pd.to_numeric(f['Temperature']) 
            # run computations using xle files
            elif filetype=='.xle':
                # open text file
                with open(infile) as fd:
                    # parse xml
                    obj = xmltodict.parse(fd.read(),encoding="ISO-8859-1")
                # navigate through xml to the data
                wellrawdata = obj['Body_xle']['Data']['Log']
                # convert xml data to pandas dataframe
                f[transport.getfilename(infile)] = pd.DataFrame(wellrawdata)
                # get header names and apply to the pandas dataframe          
                f[transport.getfilename(infile)][str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f[transport.getfilename(infile)]['ch1']
                f[transport.getfilename(infile)][str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f[transport.getfilename(infile)]['ch2']
      
                # add extension-free file name to dataframe
                f[transport.getfilename(infile)]['name'] = transport.getfilename(infile)
                # combine Date and Time fields into one field
                f[transport.getfilename(infile)]['DateTime'] = pd.to_datetime(f[transport.getfilename(infile)].apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
                f[transport.getfilename(infile)] = f[transport.getfilename(infile)].reset_index()
                f[transport.getfilename(infile)] = f[transport.getfilename(infile)].set_index('DateTime')
                f[transport.getfilename(infile)] = f[transport.getfilename(infile)].drop(['Date','Time','@id','ch1','ch2','index','ms'],axis=1)
            # run computations using csv files
    
            else:
                pass
        # concatenate all of the DataFrames in dictionary f to one DataFrame: g
        g = pd.concat(f)
        # remove multiindex and replace with index=Datetime
        g = g.reset_index()
        g = g.set_index(['DateTime'])
        # drop old indexes
        g = g.drop(['level_0'],axis=1)
        # remove duplicates based on index then sort by index
        g['ind']=g.index
        g.drop_duplicates(subset='ind',inplace=True)
        g.drop('ind',axis=1,inplace=True)
        g = g.sort()
        outfile = g
        return outfile
    
    @staticmethod 
    def appendomatic(infile,existingfile):
        '''
        appends data from one table to an existing compilation
        this tool will delete and replace the existing file
    
        infile = input file
        existingfile = file you wish to append to
        '''
    
        # get the extension of the input file
        filetype = os.path.splitext(infile)[1]
        
        # run computations using lev files
        if filetype=='.lev':
            # open text file
            with open(infile) as fd:
                # find beginning of data
                indices = fd.readlines().index('[Data]\n')
    
            # convert data to pandas dataframe starting at the indexed data line
            f = pd.read_table(infile, parse_dates=True, sep='     ', index_col=0,
                                           skiprows=indices+2, names=['DateTime','Level','Temperature'], skipfooter=1,engine='python')
            # add extension-free file name to dataframe
            f['name'] = transport.getfilename(infile)
    
        # run computations using xle files
        elif filetype=='.xle':
            f = transport.new_xle_imp(infile)
        
        # run computations using csv files
        elif filetype=='.csv':
            with open(infile) as fd:
            # find beginning of data
                try:
                    indices = fd.readlines().index('Date,Time,ms,Level,Temperature\n')
                except ValueError:
                    indices = fd.readlines().index(',Date,Time,100 ms,Level,Temperature\n')
            f = pd.read_csv(infile, skiprows=indices, skipfooter=1, engine='python')
            # add extension-free file name to dataframe
            f['name'] = transport.getfilename(infile)
            # combine Date and Time fields into one field
            f['DateTime'] = pd.to_datetime(f.apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
            f = f.reset_index()
            f = f.set_index('DateTime')
            f = f.drop(['Date','Time','ms','index'],axis=1)
                # skip other file types
        else:
            pass
    
        # ensure that the Level and Temperature data are in a float format
        f['Level'] = pd.to_numeric(f['Level'])
        f['Temperature'] = pd.to_numeric(f['Temperature'])
        h = pd.read_csv(existingfile,index_col=0,header=0,parse_dates=True)
        g = pd.concat([h,f])
        # remove duplicates based on index then sort by index
        g['ind']=g.index
        g.drop_duplicates(subset='ind',inplace=True)
        g.drop('ind',axis=1,inplace=True)
        g = g.sort_index()    
        os.remove(existingfile)
        g.to_csv(existingfile)
    
    @staticmethod
    def make_files_table(folder, wellinfo):
        '''
        This tool will make a descriptive table (Pandas DataFrame) containing filename, date, and site id.
        For it to work properly, files must be named in the following fashion:
        siteid YYYY-MM-DD
        example: pw03a 2015-03-15.csv
    
        This tool assumes that if there are two files with the same name but different extensions, 
        then the datalogger for those data is a Solinst datalogger.
    
        folder = directory containing the newly downloaded transducer data
        '''
    
        filenames = next(os.walk(folder))[2]
        site_id, exten, dates, fullfilename = [],[],[],[]
        # parse filename into relevant pieces
        for i in filenames:
            site_id.append(i[:-14].lower().strip())
            exten.append(i[-4:])
            dates.append(i[-14:-4])
            fullfilename.append(i)
        files = {'siteid':site_id,'extensions':exten,'date':dates,'full_file_name':fullfilename}
        files = pd.DataFrame(files)
        #files['filedups'] = files.duplicated(subset='siteid')
        #files['LoggerTypeID'] = files['filedups'].astype('int')+1
        files['LoggerTypeName']=files['extensions'].apply(lambda x: 'Global Water' if x=='.csv' else 'Solinst',1)
        files.drop_duplicates(subset='siteid',keep='last',inplace=True)
    
        #wellinfo = pd.read_csv(wellinfofile,header=0,index_col=0)
        wellinfo = wellinfo[wellinfo['Well']!=np.nan]
        wellinfo["G_Elev_m"] = np.divide(wellinfo["GroundElevation"],3.2808)
        wellinfo['Well'] = wellinfo['Well'].apply(lambda x: str(x).lower().strip())
        files = pd.merge(files,wellinfo,left_on='siteid',right_on='Well')
        
        return files
    
    @staticmethod
    def barodistance(wellinfo):
        '''
        Determines Closest Barometer to Each Well using wellinfo DataFrame
        '''
        barometers = {'barom':['pw03','pw10','pw19'], 'X':[240327.49,271127.67,305088.9], 
                      'Y':[4314993.95,4356071.98,4389630.71], 'Z':[1623.079737,1605.187759,1412.673738]}
        barolocal = pd.DataFrame(barometers)
        barolocal = barolocal.reset_index()
        barolocal.set_index('barom',inplace=True)
    
        wellinfo['pw03'] = np.sqrt((barolocal.loc['pw03','X']-wellinfo['UTMEasting'])**2 + 
                                       (barolocal.loc['pw03','Y']-wellinfo['UTMNorthing'])**2 + 
                                       (barolocal.loc['pw03','Z']-wellinfo['G_Elev_m'])**2)
        wellinfo['pw10'] = np.sqrt((barolocal.loc['pw10','X']-wellinfo['UTMEasting'])**2 + 
                                       (barolocal.loc['pw10','Y']-wellinfo['UTMNorthing'])**2 + 
                                       (barolocal.loc['pw10','Z']-wellinfo['G_Elev_m'])**2)
        wellinfo['pw19'] = np.sqrt((barolocal.loc['pw19','X']-wellinfo['UTMEasting'])**2 + 
                                       (barolocal.loc['pw19','Y']-wellinfo['UTMNorthing'])**2 + 
                                       (barolocal.loc['pw19','Z']-wellinfo['G_Elev_m'])**2)
        wellinfo['closest_baro'] = wellinfo[['pw03','pw10','pw19']].T.idxmin()
        return wellinfo   
    
    @staticmethod
    def imp_new_well(infile, wellinfo, manual, baro):
        '''
        INPUT
		-----
        infile = full file path of well to import
        wellinfo = Pandas DataFrame containing information of wells
        manual = Pandas DataFrame containing manual water level measurements
        
        RETURNS
		-------
        A Pandas DataFrame, maximum calculated drift, well name
        
        This function imports xle (Solinst) and csv (Global Water) transducer files, removes barometric pressure effects and corrects for drift.
		
		
        ''' 
        wellname, wellid = transport.getwellid(infile,wellinfo) #see custom getwellid function
        print('Well = ' + wellname)    
        if wellinfo[wellinfo['Well']==wellname]['LoggerTypeName'].values[0] == 'Solinst': # Reads Solinst Raw File
            f = transport.new_xle_imp(infile)
            # Remove first and/or last measurements if the transducer was out of the water
            f = transport.dataendclean(f,'Level')      
            
            bse = int(f.index.to_datetime().minute[0])
            try:
                bp = str(wellinfo[wellinfo['Well']==wellname]['BE barologger'].values[0])
                b = hourly_resample(baro[bp], bse)
                b = b.to_frame()
            except (KeyError,NameError):
                print('No BP match, using pw03')
                b = transport.hourly_resample(baro['pw03'], bse)
                b = b.to_frame()
                b['bp'] = b['pw03']
                bp = 'bp'
            f = transport.hourly_resample(f,bse)
            g = pd.merge(f,b,left_index=True,right_index=True,how='inner')
            
            g['MeasuredLevel'] = g['Level']         
            
            glist = f.columns.tolist()
            if 'Temperature' in glist:
                g['Temp'] = g['Temperature']
                g.drop(['Temperature'],inplace=True,axis=1)
            elif 'Temp' in glist:
                pass
            # Get Baro Efficiency
            be = wellinfo[wellinfo['WellID']==wellid]['BaroEfficiency']
            be = be.iloc[0]
    
            # Barometric Efficiency Correction
            print('Efficiency = '+str(be))
            g['BaroEfficiencyLevel'] = g[['MeasuredLevel',bp]].apply(lambda x: x[0] - x[1] + x[1]*float(be), 1)
        else: # Reads Global Water Raw File
            f = pd.read_csv(infile,skiprows=1,parse_dates=[[0,1]])
            #f = f.reset_index()
            f['DateTime'] = pd.to_datetime(f['Date_ Time'],errors='coerce')
            f = f[f.DateTime.notnull()]
            if ' Feet' in list(f.columns.values):
                f['Level'] = f[' Feet']
                f.drop([' Feet'],inplace=True,axis=1)        
            elif 'Feet' in list(f.columns.values):
                f['Level'] = f['Feet']
                f.drop(['Feet'],inplace=True,axis=1)        
            else:
                f['Level'] = f.iloc[:,1]
            # Remove first and/or last measurements if the transducer was out of the water
            f = transport.dataendclean(f,'Level')      
            flist = f.columns.tolist()
            if ' Temp C' in flist:
                f['Temperature'] = f[' Temp C']
                f['Temp'] = f['Temperature']
                f.drop([' Temp C','Temperature'],inplace=True,axis=1)
            else:
                f['Temp'] = np.nan
            f.set_index(['DateTime'], inplace=True)
            f['date'] = f.index.to_julian_date().values
            f['datediff'] = f['date'].diff()
            f = f[f['datediff']>0]
            f = f[f['datediff']<1]
            bse = int(f.index.to_datetime().minute[0])
            f = transport.hourly_resample(f,bse)
            f.drop([u' Volts',u'date',u'datediff'],inplace=True,axis=1)        
 
            try:
                bp = str(wellinfo[wellinfo['Well']==wellname]['BE barologger'].values[0])
                b = transport.hourly_resample(baro[bp], bse)
                b = b.to_frame()
            except (KeyError,NameError):
                print('No match, using Level')
                bp = u'Level'
                b = b.to_frame()
                b['bp'] = b['Level']
                bp = 'bp'
                b.drop(['bp'],inplace=True,axis=1)
                
            #b = hourly_resample(baro[bp], bse)
            f = transport.hourly_resample(f,bse)
            g = pd.merge(f,b,left_index=True,right_index=True,how='inner')
            g['MeasuredLevel'] = g['Level']
            
            # Get Baro Efficiency
            be = wellinfo[wellinfo['WellID']==wellid]['BaroEfficiency']
            be = be.iloc[0]
        
            # Barometric Efficiency Correction
            #g['BaroEfficiencyCorrected'] = g['MeasuredLevel'] + be*g[bp]
            print('Efficiency = '+str(be))
            g['BaroEfficiencyLevel'] = g[['MeasuredLevel',str(bp)]].apply(lambda x: x[0] + x[1]*float(be), 1)
                    
        g['DeltaLevel'] = g['BaroEfficiencyLevel'] - g['BaroEfficiencyLevel'][0]
        
        # Match manual water level to closest date
        g['MeasuredDTW'] = transport.fcl(manual[manual['WellID']== wellid],min(g.index.to_datetime()))[1]-g['DeltaLevel']
    
        # Drift Correction
        #lastdtw = g['MeasuredDTW'][-1]
        last = transport.fcl(manual[manual['WellID']== wellid],max(g.index.to_datetime()))[1]
        first = transport.fcl(manual[manual['WellID']== wellid],min(g.index.to_datetime()))[1]
        lastg = float(g[g.index.to_datetime()==max(g.index.to_datetime())]['MeasuredDTW'].values)
        driftlen = len(g.index)
        g['last_diff_int'] = np.round((lastg-last),4)/np.round(driftlen-1.0,4)
        g['DriftCorrection'] = np.round(g['last_diff_int'].cumsum()-g['last_diff_int'],4)
        print('Max Drift = '+str(g['DriftCorrection'][-1]))
        # Assign well id to column
        g['WellID'] = wellid
        
        # Get Depth to water below casing
        g['DTWBelowCasing'] = g['MeasuredDTW'] - g['DriftCorrection']
    
        # subtract casing height from depth to water below casing
        g['DTWBelowGroundSurface'] = g['DTWBelowCasing'] - wellinfo[wellinfo['WellID']==wellid]['Offset'].values[0]
        
        # subtract depth to water below ground surface from well surface elevation
        g['WaterElevation'] = wellinfo[wellinfo['WellID']==wellid]['GroundElevation'].values[0] - g['DTWBelowGroundSurface']
        g['WaterElevation'] = g['WaterElevation'].apply(lambda x: round(x,2))
        # assign tape value
        g['Tape'] = 0
        g['MeasuredByID'] = 0
        
        g['DateTime'] = g.index.to_datetime()
        g = g.loc[:,["WellID","DateTime","MeasuredLevel","Temp","BaroEfficiencyLevel","DeltaLevel", 
                     "MeasuredDTW","DriftCorrection","DTWBelowCasing","DTWBelowGroundSurface",
                     "WaterElevation","Tape","MeasuredByID"]]
        maxDrift = g['DriftCorrection'][-1]
        return g, maxDrift, wellname
    
    
    # Use `g[wellinfo[wellinfo['Well']==wellname]['closest_baro']]` instead if you want to match the closest barometer to the data

	@staticmethod
	def manualset(wellbaro, meas, manualfile, manmeas, outcolname = 'corr_wl'):
	    '''
	    INPUT
	    -----
	    wellbaro = Pandas DataFrame containing aligned Barometric and Level data
	    meas = name of column in wellbaro dataframe that contains transducer water level data
	    
	    manualfile = Pandas DataFrame containing manual measurements
	    manmeas = name of column in manualfile dataframe that contains manual measurements
	    
	    outcolname = name of output column for corrected data
	    
	    manual and transducer water level measurements should be in the same units
	    
	    RETURNS
	    -------
	    dataframe with outcolname field
	    '''
	    breakpoints = []
	    bracketedwls = {}
	    dtnm = wellbaro.index.name
	    manualfile['julian'] = manualfile.index.to_julian_date() 
	
	    for i in range(len(manualfile)):
	        breakpoints.append(transport.fcl(wellbaro, manualfile.index.to_datetime()[i]).name)
	    breakpoints = sorted(list(set(breakpoints)))
	
	    for i in range(len(breakpoints)-1):
	        # Break up pandas dataframe time series into pieces based on timing of manual measurements
	        bracketedwls[i] = wellbaro.loc[(wellbaro.index.to_datetime() > breakpoints[i])&(wellbaro.index.to_datetime() < breakpoints[i+1])]
	        if len(bracketedwls[i]) > 0:
	            bracketedwls[i].loc[:,'julian'] = bracketedwls[i].index.to_julian_date()
	    
	            last_man = transport.fcl(manualfile,breakpoints[i+1])
	            first_man = transport.fcl(manualfile,breakpoints[i])
	            b = first_man[manmeas] - bracketedwls[i].ix[0, meas]
	            m = (last_man[manmeas] - first_man[manmeas])/(last_man['julian'] - first_man['julian'])
	            bracketedwls[i].loc[:,'datechange'] = bracketedwls[i].ix[:,'julian'] - bracketedwls[i].ix[0,'julian']
	            bracketedwls[i].loc[:,'wldiff'] = bracketedwls[i].loc[:,meas]-bracketedwls[i].ix[0,meas]
	            bracketedwls[i].loc[:,outcolname] = bracketedwls[i][['datechange',meas]].apply(lambda x: x[1]+(m*x[0]+b), 1)
	        else:
	            pass
	    wellbarofixed = pd.concat(bracketedwls)
	    wellbarofixed.reset_index(inplace=True)
	    wellbarofixed.set_index(dtnm ,inplace=True)
	
	    return wellbarofixed
    
    @staticmethod
    def smoother(df, p, win=30, sd=3):
        '''
        remove outliers from a pandas dataframe column and fill with interpolated values
        warning: this will fill all NaN values in the dataframe with the interpolate function
        
        INPUT
        ------
        df= dataframe of interest
        p= column in dataframe with outliers
        win= size of window
        std= number of standard deviations allowed
        
        RETURNS
        ------
        Pandas dataframe with outliers removed
        '''
        df1 = df
        df1.loc[:,'dp'+ p] = df1.loc[:,p].diff()
        df1.loc[:,'ma'+ p] = pd.rolling(df1.loc[:,'dp'+ p], window=win, center=True).mean()
        df1.loc[:,'mst'+p] = pd.rolling(df1.loc[:,'dp'+ p], window=win, center=True).std()
        for i in df.index:
            try:
                if abs(df1.loc[i,'dp'+ p] - df1.loc[i,'ma'+ p]) >= abs(df1.loc[i,'mst'+p]*sd):
                    df.loc[i,p]=np.nan
                else:
                    df.loc[i,p]=df.loc[i,p]
            except (ValueError):
                try:
                    if abs(df1.loc[i,'dp'+ p] - df1.loc[i,'ma'+ p]) >= abs(df1.loc[:,'dp'+p].std()*sd):
                        df.loc[i,p]=np.nan
                    else:
                        df.loc[i,p]=df.loc[i,p]
                except (ValueError):
                    df.loc[i,p]=df.loc[i,p]
    
        try:
            df1 = df1.drop(['dp'+p,'ma'+p,'mst'+p],axis=1)
        except(NameError,ValueError):
            pass            
        del df1
        try:
            df = df.drop(['dp'+p,'ma'+p,'mst'+p],axis=1)
        except(NameError,ValueError):
            pass  
        df = df.interpolate(method='time', limit=30)
        df = df[1:-1]
        return df
        
    @staticmethod
    def baro_drift_correct(wellfile,barofile, manualfile, sampint=60,wellelev=4800,stickup=0):
        '''
        INPUT
        -----
        wellfile = Pandas DataFrame with water level data labeled 'Level'; index must be datetime
        barofile = Pandas DataFrame with barometric data labeled 'Level'; index must be datetime
        manualfile = Pandas DataFrame with manual level data in the first column after the index; index must be datetime
        
        sampint = sampling interval in minutes; default 60
        wellelev = site ground surface elevation in feet
        stickup = offset of measure point from ground in feet
        
        OUTPUT
        -----
        wellbarofinal = pandas dataframe with corrected water levels 
        
        This function uses pandas dataframes created using the 
    
        '''
        #Remove dangling ends
        baroclean = transport.dataendclean(barofile, 'Level')
        wellclean = transport.dataendclean(wellfile, 'Level')
        
        # resample data to make sample interval consistent  
        baro = transport.hourly_resample(baroclean,0,sampint)
        well = transport.hourly_resample(wellclean,0,sampint)
        
        # reassign `Level` to reduce ambiguity
        well['abs_feet_above_levelogger'] = well['Level']
        baro['abs_feet_above_barologger'] = baro['Level']
        
        # combine baro and well data for easy calculations, graphing, and manipulation
        wellbaro = pd.merge(well,baro,left_index=True,right_index=True,how='inner')
        wellbaro['adjusted_levelogger'] =  wellbaro['abs_feet_above_levelogger'] - wellbaro['abs_feet_above_barologger']
        
        breakpoints = []
        bracketedwls = {}
    
        for i in range(len(manualfile)+1):
            breakpoints.append(transport.fcl(wellbaro, manualfile.index.to_datetime()[i-1]).name)
    
        last_man_wl,first_man_wl,last_tran_wl,driftlen = [],[],[],[]
       
        for i in range(len(manualfile)-1):
            # Break up time series into pieces based on timing of manual measurements
            bracketedwls[i+1] = wellbaro.loc[(wellbaro.index.to_datetime() > breakpoints[i+1])&(wellbaro.index.to_datetime() < breakpoints[i+2])]
            bracketedwls[i+1]['diff_wls'] = bracketedwls[i+1]['abs_feet_above_levelogger'].diff() 
    
    
            bracketedwls[i+1].loc[:,'DeltaLevel'] = bracketedwls[i+1].loc[:,'adjusted_levelogger'] - bracketedwls[i+1].ix[0,'adjusted_levelogger']
            bracketedwls[i+1].loc[:,'MeasuredDTW'] = transport.fcl(manualfile,breakpoints[i+1])[0] - bracketedwls[i+1].loc[:,'DeltaLevel']
    
            last_man_wl.append(transport.fcl(manualfile,breakpoints[i+2])[0])
            first_man_wl.append(transport.fcl(manualfile,breakpoints[i+1])[0])
            last_tran_wl.append(float(bracketedwls[i+1].loc[max(bracketedwls[i+1].index.to_datetime()),'MeasuredDTW']))
            driftlen.append(len(bracketedwls[i+1].index))
            bracketedwls[i+1].loc[:,'last_diff_int'] = np.round((last_tran_wl[i]-last_man_wl[i]),4)/np.round(driftlen[i]-1.0,4)
            bracketedwls[i+1].loc[:,'DriftCorrection'] = np.round(bracketedwls[i+1].loc[:,'last_diff_int'].cumsum()-bracketedwls[i+1].loc[:,'last_diff_int'],4)
    
        wellbarofixed = pd.concat(bracketedwls)
        wellbarofixed.reset_index(inplace=True)
        wellbarofixed.set_index('DateTime',inplace=True)
        # Get Depth to water below casing
        wellbarofixed.loc[:,'DTWBelowCasing'] = wellbarofixed['MeasuredDTW'] - wellbarofixed['DriftCorrection']
    
        # subtract casing height from depth to water below casing
        wellbarofixed.loc[:,'DTWBelowGroundSurface'] = wellbarofixed.loc[:,'DTWBelowCasing'] - stickup #well riser height
    
        # subtract depth to water below ground surface from well surface elevation
        wellbarofixed.loc[:,'WaterElevation'] = wellelev - wellbarofixed.loc[:,'DTWBelowGroundSurface']
        
        wellbarofinal = transport.smoother(wellbarofixed, 'WaterElevation')
        
        return wellbarofinal
    
    @staticmethod
    def xleHeadTable(folder):
        '''
		Creates a Pandas Dataframe containing header information from all xle files in a folder
		
		INPUT
		-----
		folder = folder containing xle files
		
        RETURNS
        -------
		A Pandas DataFrame containing the transducer data
        '''
        # open text file
        
        filenames = [os.path.join(folder,fn) for fn in next(os.walk(folder))[2]]
        
        instType, modelNum, serialNum, firmWare, project, well, stopTime, batteryPct = [],[],[],[],[],[],[],[] 
        
        for infile in filenames:
           
            # get the extension of the input file
            filename, filetype = os.path.splitext(infile)
            if filetype=='.xle':
                # open text file
                with open(infile) as fd:
                    # parse xml
                    obj = xmltodict.parse(fd.read(),encoding="ISO-8859-1")
                # navigate through xml to the data
                instType.append(obj['Body_xle']['Instrument_info']['Instrument_type'])          
                modelNum.append(obj['Body_xle']['Instrument_info']['Model_number'])  
                serialNum.append(obj['Body_xle']['Instrument_info']['Serial_number'])
                batteryPct.append(obj['Body_xle']['Instrument_info']['Battery_level'])
                firmWare.append(obj['Body_xle']['Instrument_info']['Firmware'])
                project.append(obj['Body_xle']['Instrument_info_data_header']['Project_ID'])
                well.append(obj['Body_xle']['Instrument_info_data_header']['Location'])
                stopTime.append(obj['Body_xle']['Instrument_info_data_header']['Stop_time'])
        properties = pd.DataFrame({'instType':instType, 'modelNum':modelNum, 'serialNum':serialNum, 'firmWare':firmWare, 
                                   'project':project, 'well':well, 'stopTime':stopTime, 'batteryPct':batteryPct})
    
        return properties
