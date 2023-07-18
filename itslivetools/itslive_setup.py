import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
#itslive classes and functions

class IndGlacier:
    
    def __init__(self, rgi_id, rgi_outline, dem, itslive, centerLine=None, ablationLine = None, lowestPoint = None):
        
        self.rgi_id = rgi_id
        self.rgi_outline = rgi_outline
        self.dem = dem
        self.itslive = itslive
        self.centerLine = centerLine
        self.ablationLine = ablationLine
        self.lowestPoint = lowestPoint
    
 

def ind_glacier_data_prep(rgi_id, rgi_full, itslive_dc, dem_obj, centerlines, ablationlines, lowestpoints, utm_code):
    '''function to prepare data to create an object of the `IndGlacier` class for a single glacier.
    Pass in RGI ID of glacier of interest as well as full data objects of RGI gpdf, itslive datacube, 
    nasadem and the u tm code fo the glacier.I feel like at scale this is probably a really inefficient way to build objects'''
    
    #clip rgi to glacier
    single_rgi = rgi_full.loc[rgi_full['RGIId'] == rgi_id]
    #extract glims id - will use to extract centerlines
    glims_id = single_rgi['GLIMSId'].values
    
    
    #clip dem and itslive
    dem_clip = dem_obj.rio.clip(single_rgi.geometry, dem_obj.rio.crs).squeeze().transpose()
    itslive_clip = itslive_dc.rio.clip(single_rgi.geometry, itslive_dc.rio.crs)
    dem_clip_downsamp = dem_clip.interp_like(itslive_clip, method = 'nearest')
    itslive_clip['z'] = dem_clip_downsamp.Band1
    
    #centerline objects
    centerline = centerlines.loc[centerlines['GLIMS_ID'] == glims_id[0]].to_crs(utm_code)
    ablationline = ablationlines.loc[ablationlines['GLIMS_ID'] == glims_id[0]].to_crs(utm_code)
    lowestpoint = lowestpoints.loc[lowestpoints['GLIMS_ID'] == glims_id[0]].to_crs(utm_code)
    
    #flowline object
    
    #centerline = centerline.to_crs(utm_code)
    #ablationline = ablationline.to_crs(utm_code)
    #lowestpoint = lowestpoints.to_crs(
    
    #creat object of IndGlacier class
    rgi_outline = single_rgi
    glacier = IndGlacier(rgi_id, rgi_outline, dem_clip_downsamp, itslive_clip, centerline, ablationline, lowestpoint)
    
    return glacier