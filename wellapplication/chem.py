# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 09:50:51 2016

@author: paulinkenbrandt
"""

import pandas as pd
import numpy as np
import datetime

class WQP:
    ''' 
    WQP manipulates and imports data from this site: <a href=http://www.waterqualitydata.us/>WQP</a> 
    '''
    @staticmethod
    def datetimefix(x,format):
        '''
        This script cleans date-time errors
        
        input
        x = date-time string
        format = format of date-time string
        
        output 
        formatted datetime type
        '''
        d = str(x[0]).lstrip().rstrip()[0:10]
        t = str(x[1]).lstrip().rstrip()[0:5].zfill(5)
        try:
            int(d[0:2])
        except(ValueError,TypeError,NameError):
            return np.nan
        try:
            int(t[0:2])
            int(t[3:5])
        except(ValueError,TypeError,NameError):
            t = "00:00"
       
        if int(t[0:2])>23:
            t = "00:00"
        elif int(t[3:5])>59:
            t = "00:00"
        else:
            t = t[0:2].zfill(2) + ":" + t[3:5]
        return datetime.datetime.strptime(d + " " + t, format)    
    
    @staticmethod
    def parnorm(x):
        p = str(x[0]).rstrip().lstrip().lower()
        u = str(x[2]).rstrip().lstrip().lower()
        if p == 'nitrate' and u == 'mg/l as n':
            return 'Nitrate', x[1]*4.427, 'mg/l'
        elif p == 'nitrite' and u == 'mg/l as n':
            return 'Nitrite', x[1]*3.285, 'mg/l'
        elif p == 'ammonia-nitrogen' or p == 'ammonia-nitrogen as n' or p == 'ammonia and ammonium':
            return 'Ammonium', x[1]*1.288, 'mg/l'
        elif p == 'ammonium' and u == 'mg/l as n':
            return 'Ammonium', x[1]*1.288, 'mg/l'
        elif p == 'sulfate as s':
            return 'Sulfate', x[1]*2.996, 'mg/l'
        elif p in ('phosphate-phosphorus', 'phosphate-phosphorus as p','orthophosphate as p'):
            return 'Phosphate', x[1]*3.066, 'mg/l'
        elif (p == 'phosphate' or p == 'orthophosphate') and u == 'mg/l as p':
            return 'Phosphate', x[1]*3.066, 'mg/l'
        elif u == 'ug/l':
            return x[0], x[1]/1000, 'mg/l'
        else:
            return x[0], x[1], str(x[2]).rstrip()    
    
    @staticmethod
    def unitfix(x):
        z = str(x).lower()
        if z == "ug/l":
            return "mg/l"
        elif z == "mg/l":
            return "mg/l"
        else:
            return x    
    
    
    @staticmethod
    def WQPimportRes(csv):
        '''
        Bring data from WQP site into a Pandas Dataframe for analysis
        
        INPUT
        -----
        csv = path to csv file containing WQP data download
        
        RETURNS
        -------
        df = dataframe containing WQP data
        '''
        
        # set data types
        Rdtypes = {"OrganizationIdentifier":np.str_, "OrganizationFormalName":np.str_, "ActivityIdentifier":np.str_, 
               "ActivityStartTime/Time":np.str_,
               "ActivityTypeCode":np.str_, "ActivityMediaName":np.str_, "ActivityMediaSubdivisionName":np.str_, 
               "ActivityStartDate":np.str_, "ActivityStartTime/Time":np.str_, "ActivityStartTime/TimeZoneCode":np.str_, 
               "ActivityEndDate":np.str_, "ActivityEndTime/Time":np.str_, "ActivityEndTime/TimeZoneCode":np.str_, 
               "ActivityDepthHeightMeasure/MeasureValue":np.float16, "ActivityDepthHeightMeasure/MeasureUnitCode":np.str_, 
               "ActivityDepthAltitudeReferencePointText":np.str_, "ActivityTopDepthHeightMeasure/MeasureValue":np.float16, 
               "ActivityTopDepthHeightMeasure/MeasureUnitCode":np.str_, 
               "ActivityBottomDepthHeightMeasure/MeasureValue":np.float16, 
               "ActivityBottomDepthHeightMeasure/MeasureUnitCode":np.str_, 
               "ProjectIdentifier":np.str_, "ActivityConductingOrganizationText":np.str_, 
               "MonitoringLocationIdentifier":np.str_, "ActivityCommentText":np.str_, 
               "SampleAquifer":np.str_, "HydrologicCondition":np.str_, "HydrologicEvent":np.str_, 
               "SampleCollectionMethod/MethodIdentifier":np.str_, "SampleCollectionMethod/MethodIdentifierContext":np.str_, 
               "SampleCollectionMethod/MethodName":np.str_, "SampleCollectionEquipmentName":np.str_, 
               "ResultDetectionConditionText":np.str_, "CharacteristicName":np.str_, "ResultSampleFractionText":np.str_, 
               "ResultMeasureValue":np.str_, "ResultMeasure/MeasureUnitCode":np.str_, "MeasureQualifierCode":np.str_, 
               "ResultStatusIdentifier":np.str_, "StatisticalBaseCode":np.str_, "ResultValueTypeName":np.str_, 
               "ResultWeightBasisText":np.str_, "ResultTimeBasisText":np.str_, "ResultTemperatureBasisText":np.str_, 
               "ResultParticleSizeBasisText":np.str_, "PrecisionValue":np.str_, "ResultCommentText":np.str_, 
               "USGSPCode":np.str_, "ResultDepthHeightMeasure/MeasureValue":np.float16, 
               "ResultDepthHeightMeasure/MeasureUnitCode":np.str_, "ResultDepthAltitudeReferencePointText":np.str_, 
               "SubjectTaxonomicName":np.str_, "SampleTissueAnatomyName":np.str_, 
               "ResultAnalyticalMethod/MethodIdentifier":np.str_, "ResultAnalyticalMethod/MethodIdentifierContext":np.str_, 
               "ResultAnalyticalMethod/MethodName":np.str_, "MethodDescriptionText":np.str_, "LaboratoryName":np.str_, 
               "AnalysisStartDate":np.str_, "ResultLaboratoryCommentText":np.str_, 
               "DetectionQuantitationLimitTypeName":np.str_, "DetectionQuantitationLimitMeasure/MeasureValue":np.str_, 
               "DetectionQuantitationLimitMeasure/MeasureUnitCode":np.str_, "PreparationStartDate":np.str_, 
               "ProviderName":np.str_} 

        # define date field indices
        dt = [6,56,61]
        
        # read csv into DataFrame
        df = pd.read_csv(csv, dtype=Rdtypes, parse_dates=dt)
        return df        
    
    @staticmethod
    def WQPmassageResults(df):
        '''
        Massage WQP result data for analysis
        
        INPUT
        -----
        df = dataframe containing raw WQP data
        
        RETURNS
        -------
        df = dataframe containing cleaned up WQP data
        '''
        # Map new names for columns
        ResFieldDict = {"AnalysisStartDate":"AnalysisDate", "ResultAnalyticalMethod/MethodIdentifier":"AnalytMeth", 
                "ResultAnalyticalMethod/MethodName":"AnalytMethId", "ResultDetectionConditionText":"DetectCond", 
                "ResultLaboratoryCommentText":"LabComments", "LaboratoryName":"LabName", 
                "DetectionQuantitationLimitTypeName":"LimitType", "DetectionQuantitationLimitMeasure/MeasureValue":"MDL", 
                "DetectionQuantitationLimitMeasure/MeasureUnitCode":"MDLUnit", "MethodDescriptionText":"MethodDescript", 
                "OrganizationIdentifier":"OrgId", "OrganizationFormalName":"OrgName", "CharacteristicName":"Param", 
                "ProjectIdentifier":"ProjectId", "MeasureQualifierCode":"QualCode", "ResultCommentText":"ResultComment", 
                "ResultStatusIdentifier":"ResultStatus", "ResultMeasureValue":"ResultValue", 
                "ActivityCommentText":"SampComment", "ActivityDepthHeightMeasure/MeasureValue":"SampDepth", 
                "ActivityDepthAltitudeReferencePointText":"SampDepthRef", 
                "ActivityDepthHeightMeasure/MeasureUnitCode":"SampDepthU", "SampleCollectionEquipmentName":"SampEquip", 
                "ResultSampleFractionText":"SampFrac", "ActivityStartDate":"SampleDate", "ActivityIdentifier":"SampleId", 
                "ActivityStartTime/Time":"SampleTime", "ActivityMediaSubdivisionName":"SampMedia", 
                "SampleCollectionMethod/MethodIdentifier":"SampMeth", "SampleCollectionMethod/MethodName":"SampMethName", 
                "ActivityTypeCode":"SampType", "MonitoringLocationIdentifier":"StationId", 
                "ResultMeasure/MeasureUnitCode":"Unit", "USGSPCode":"USGSPCode",
                "ActivityStartDate":"StartDate","ActivityStartTime/Time":"StartTime"} 
                
        # Rename Data
        df1 = df.rename(columns=ResFieldDict)
        
        # Remove unwanted and bad times        
        df1["SampleDate"] = df1[["StartDate","StartTime"]].apply(lambda x: WQP.datetimefix(x,"%Y-%m-%d %H:%M"),1)
        
        # Define unneeded fields to drop
        resdroplist = ["ActivityBottomDepthHeightMeasure/MeasureUnitCode", "ActivityBottomDepthHeightMeasure/MeasureValue", 
               "ActivityConductingOrganizationText", "ActivityEndDate", "ActivityEndTime/Time", 
               "ActivityEndTime/TimeZoneCode", "ActivityMediaName", "ActivityStartTime/TimeZoneCode", 
               "ActivityTopDepthHeightMeasure/MeasureUnitCode", "ActivityTopDepthHeightMeasure/MeasureValue", 
               "HydrologicCondition", "HydrologicEvent", "PrecisionValue", "PreparationStartDate", "ProviderName", 
               "ResultAnalyticalMethod/MethodIdentifierContext", "ResultDepthAltitudeReferencePointText", 
               "ResultDepthHeightMeasure/MeasureUnitCode", "ResultDepthHeightMeasure/MeasureValue", 
               "ResultParticleSizeBasisText", "ResultTemperatureBasisText", 
               "ResultTimeBasisText", "ResultValueTypeName", "ResultWeightBasisText", "SampleAquifer", 
               "SampleCollectionMethod/MethodIdentifierContext", "SampleTissueAnatomyName", "StatisticalBaseCode", 
               "SubjectTaxonomicName","StartTime","StartDate","StartTime","StartDate"] 
        
        # Drop fields
        df1 = df1.drop(resdroplist, axis=1)
        
        # convert results and mdl to float
        df1['ResultValue'] = pd.to_numeric(df1['ResultValue'], errors='coerce')
        df1['MDL'] = pd.to_numeric(df1['MDL'], errors='coerce')
        
        # match old and new station ids
        df1['StationId'] = df1['StationId'].str.replace('_WQX-','-')
                
        #standardize all ug/l data to mg/l
        df1.Unit = df1.Unit.apply(lambda x: str(x).rstrip(), 1)
        df1.ResultValue = df1[["ResultValue","Unit"]].apply(lambda x: x[0]/1000 if str(x[1]).lower()=="ug/l" else x[0], 1)
        df1.Unit = df1.Unit.apply(lambda x: WQP.unitfix(x),1)
        

        df1['Param'], df1['ResultValue'], df1['Unit'] = zip(*df1[['Param','ResultValue','Unit']].apply(lambda x: WQP.parnorm(x),1))
        
        return df1
        
    @staticmethod
    def WQPmassageStations(df):
        '''
        Massage WQP station data for analysis
        
        INPUT
        -----
        df = dataframe containing raw WQP data
        
        RETURNS
        -------
        df = dataframe containing cleaned up WQP data
        '''
        StatFieldDict = {"MonitoringLocationIdentifier":"StationId", "AquiferName":"Aquifer", "AquiferTypeName":"AquiferType", 
             "ConstructionDateText":"ConstDate", "CountyCode":"CountyCode", "WellDepthMeasure/MeasureValue":"Depth", 
             "WellDepthMeasure/MeasureUnitCode":"DepthUnit", "VerticalMeasure/MeasureValue":"Elev", 
             "VerticalAccuracyMeasure/MeasureValue":"ElevAcc", "VerticalAccuracyMeasure/MeasureUnitCode":"ElevAccUnit", 
             "VerticalCollectionMethodName":"ElevMeth", "VerticalCoordinateReferenceSystemDatumName":"ElevRef", 
             "VerticalMeasure/MeasureUnitCode":"ElevUnit", "FormationTypeText":"FmType", 
             "WellHoleDepthMeasure/MeasureValue":"HoleDepth", "WellHoleDepthMeasure/MeasureUnitCode":"HoleDUnit", 
             "HorizontalAccuracyMeasure/MeasureValue":"HorAcc", "HorizontalAccuracyMeasure/MeasureUnitCode":"HorAccUnit", 
             "HorizontalCollectionMethodName":"HorCollMeth", "HorizontalCoordinateReferenceSystemDatumName":"HorRef", 
             "HUCEightDigitCode":"HUC8", "LatitudeMeasure":"Lat_Y", "LongitudeMeasure":"Lon_X", 
             "OrganizationIdentifier":"OrgId", "OrganizationFormalName":"OrgName", "StateCode":"StateCode", 
             "MonitoringLocationDescriptionText":"StationComment", "MonitoringLocationName":"StationName", 
             "MonitoringLocationTypeName":"StationType"}
             
        df.rename(columns=StatFieldDict,inplace=True)
        
        statdroplist = ["ContributingDrainageAreaMeasure/MeasureUnitCode", "ContributingDrainageAreaMeasure/MeasureValue", 
                "DrainageAreaMeasure/MeasureUnitCode", "DrainageAreaMeasure/MeasureValue", "CountryCode", "ProviderName", 
                "SourceMapScaleNumeric"]
        
        df.drop(statdroplist,inplace=True,axis=1)
        
        TypeDict = {"Stream: Canal":"Stream", "River/Stream":"Stream", 
            "Stream: Canal":"Stream", "Well: Test hole not completed as a well":"Well"}

        # Make station types in the StationType field consistent for easier summary and compilation later on.
        df.StationType = df["StationType"].apply(lambda x: TypeDict.get(x,x),1)
        df.Elev = df.Elev.apply(lambda x: np.nan if x==0.0 else round(x,1), 1)
        
        # Remove preceding WQX from StationId field to remove duplicate station data created by legacy database.
        df['StationId'] = df['StationId'].str.replace('_WQX-','-')
        df.drop_duplicates(subset=['StationId'],inplace=True)
        
        return df