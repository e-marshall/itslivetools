import geopandas as gpd
import os
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from shapely.geometry import Polygon
from shapely.geometry import Point
import json
from scipy.stats import sem

def find_granule_by_point(input_dict, input_point): #[lon,lat]
    '''Takes an input dictionary (a geojson catalog) and a point to represent AOI.
    this returns a list of the s3 urls corresponding to zarr datacubes whose footprint covers the AOI'''
    #print([input_points][0])
    
    target_granule_urls = []
    #Point(coord[0], coord[1])
    #print(input_point[0])
    #print(input_point[1])
    point_geom = Point(input_point[0], input_point[1])
    #print(point_geom)
    point_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry = [point_geom])
    for granule in range(len(input_dict['features'])):
        
        #print('tick')
        bbox_ls = input_dict['features'][granule]['geometry']['coordinates'][0]
        bbox_geom = Polygon(bbox_ls)
        bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry = [bbox_geom])
        
        #if poly_gdf.contains(points1_ls[poly]).all() == True:

        if bbox_gdf.contains(point_gdf).all() == True:
            #print('yes')
            target_granule_urls.append(input_dict['features'][granule]['properties']['zarr_url'])
        else:
            pass
            #print('no')
    return target_granule_urls

def read_in_s3(http_url, chunks = 'auto'):
    '''this function takes an http url (from itslive catalog), converts to s3 url and returns the zarr datacube stored in the s3 bucket pointed to w/ url.
    will also set the crs based on the projection included in the attrs
    '''
    s3_url = http_url.replace('http','s3')
    s3_url = s3_url.replace('.s3.amazonaws.com','')

    datacube = xr.open_dataset(s3_url, engine = 'zarr',
                                storage_options={'anon':True},
                                chunks = chunks)
    #set crs from attrs
    datacube = datacube.rio.write_crs(f"EPSG:{datacube.attrs['projection']}")
    return datacube

def calc_sem(x):
    ''' calc standard error of measurement for an xarray data array at a given time step
    '''
    return sem(((x)*365).flatten(), nan_policy='omit')


def clip_glacier_add_dem(rgi_id, rgi_full, itslive_xr, dem_xr): #all in local utm
    '''this function is the processing step to go from a full itslive granule to the extent of a single glacier and add NASADEM to itslive xr object. Currently oriented toward seasonal analysis, could change. 
    inputs are: RGIId, rgi geodataframe, itslive xr object, nasadem xr object
    Steps are:
    1. clip itslive granule by glacier outline (or centerline)
    2. calc + add 'coverage' variable to xr.dataset
    3. subset dataset, removing time steps where cov < 0.5
    5. drop time steps where img1 and img2 are not in the same season (change this if using for full time series, not seasonal means)
    6. clip NASASDEM to extent of single glacier
    7. downsample to match itslive resolution
    8. add z as var to xr object
    9. calculate elevation quartiles and add them as variables (z0,z1,z2,z3)
    '''
    
    rgi_single = rgi_full.loc[rgi_full['RGIId'] == rgi_id]
    #print(rgi_single['RGIId'])
    
    itslive_clip = itslive_xr.rio.clip(rgi_single.geometry, rgi_single.crs)
    
    valid_pixels = itslive_clip.v.count(dim=['x','y'])
    valid_pixels_max = itslive_clip.v.notnull().any('mid_date').sum(['x','y'])
    itslive_clip['cov'] = valid_pixels / valid_pixels_max
    itslive_clip = itslive_clip.where(itslive_clip.cov >= 0.5, drop=True)
    
    dem_clip = dem_xr.rio.clip(rgi_single.geometry, rgi_single.crs)
    
    dem_downsamp = dem_clip.interp_like(itslive_clip, method='nearest')
    
    itslive_clip['z'] = dem_downsamp.NASADEM_HGT
    
    zmin = np.nanmin(dem_downsamp.NASADEM_HGT.data)
    zq1 = np.nanpercentile(dem_downsamp.NASADEM_HGT.data, 25)
    zmed = np.nanmedian(dem_downsamp.NASADEM_HGT.data)
    zq3 = np.nanpercentile(dem_downsamp.NASADEM_HGT.data, 75)
    zmax = np.nanmax(dem_downsamp.NASADEM_HGT.data)

    z0 = dem_downsamp.NASADEM_HGT.where(np.logical_and(dem_downsamp.NASADEM_HGT >= zmin, dem_downsamp.NASADEM_HGT <= zq1), drop=True)
    z1 = dem_downsamp.NASADEM_HGT.where(np.logical_and(dem_downsamp.NASADEM_HGT >= zq1, dem_downsamp.NASADEM_HGT <= zmed), drop=True)
    z2 = dem_downsamp.NASADEM_HGT.where(np.logical_and(dem_downsamp.NASADEM_HGT >= zmed, dem_downsamp.NASADEM_HGT <= zq3), drop=True)
    z3 = dem_downsamp.NASADEM_HGT.where(np.logical_and(dem_downsamp.NASADEM_HGT >= zq3, dem_downsamp.NASADEM_HGT <= zmax), drop=True)
    
    #itslive_clip['sem_v'] = (('mid_date'), [calc_sem(itslive_clip.isel(mid_date=t).v.data) for t in range(len(itslive_clip.mid_date))])


    itslive_clip['z0'] = z0
    itslive_clip['z1'] = z1
    itslive_clip['z2'] = z2
    itslive_clip['z3'] = z3
    
    z0_cond_min = itslive_clip.z0.min().data >= zmin
    z0_cond_max = itslive_clip.z0.max().data < zq1+1
    z1_cond_min = itslive_clip.z1.min().data >= zq1
    z1_cond_max = itslive_clip.z1.max().data <zmed + 1
    z2_cond_min = itslive_clip.z2.min().data >= zmed
    z2_cond_max = itslive_clip.z2.max().data < zq3 + 1
    z3_cond_min = itslive_clip.z3.min().data >= zq3
    z3_cond_max = itslive_clip.z3.max().data < zmax+1
    
    cond_ls = [z0_cond_min, z0_cond_max, z1_cond_min, z1_cond_max,
               z2_cond_min, z2_cond_max, z3_cond_min, z3_cond_max]
    
    test = all(i for i in cond_ls)
    
    #remove any time steps where img 1 and img 2 span multiple seasons
    itslive_clip = itslive_clip.where(itslive_clip.acquisition_date_img1.dt.season == itslive_clip.acquisition_date_img2.dt.season, drop=True)
    
    itslive_clip_gb = itslive_clip.groupby(itslive_clip.mid_date.dt.season).mean()
    
    if test != True:
        
        print('there is an elevation masking issue here')
        
    else:
    
        pass
    
        return itslive_clip_gb

def calc_seasonal_sem_by_z(input_gb, z, rgi_id):
    '''
    calculate mean standard error of measurement over each season. for full glacier and individual glacier elevation quartiles
    use w/ list comp passing a list like ['full','z0','z1','z2','z3']

    '''
    
    if z == 'full':
        
        winter = input_gb.sel(season='DJF')['sem_v'].data
        spring = input_gb.sel(season='MAM')['sem_v'].data
        summer = input_gb.sel(season='JJA')['sem_v'].data
        fall = input_gb.sel(season='SON')['sem_v'].data
    
    else:
        
        z_gb = input_gb.where(input_gb[f'{z}'].notnull(), drop=True)
        z_gb['sem_v'] = (('season'), [calc_sem(z_gb.isel(season=s).v.data) for s in range(len(z_gb.season))])
        
        winter = z_gb.sel(season='DJF')['sem_v'].data
        spring = z_gb.sel(season='MAM')['sem_v'].data
        summer = z_gb.sel(season='JJA')['sem_v'].data
        fall = z_gb.sel(season='SON')['sem_v'].data
        
    d = {'RGIId':rgi_id, 'var':var, 'z':z, 'winter': winter,
             'spring':spring, 'summer': summer, 'fall':fall}
            
    df = pd.DataFrame(d, index=[0])
    
    return df

def calc_seasonal_mean_v_by_z(input_gb, z, var, rgi_id):
    '''calculate mean magnitude of velocity (or any other var) for each season. for full glacier area or individual glacier elevation 
    use w/ list comp passing a list like ['full','z0','z1','z2','z3']
    '''
    
    if z == 'full':
        
        winter = input_gb.sel(season='DJF')[f'{var}'].mean(dim=['x','y']).compute().data
        spring = input_gb.sel(season='MAM')[f'{var}'].mean(dim= ['x','y']).compute().data
        summer = input_gb.sel(season='JJA')[f'{var}'].mean(dim= ['x','y']).compute().data
        fall = input_gb.sel(season='SON')[f'{var}'].mean(dim= ['x','y']).compute().data
        
    else:
        z_gb = input_gb.where(input_gb[f'{z}'].notnull(), drop=True)
        
        winter = z_gb.sel(season='DJF')[f'{var}'].mean(dim=['x','y']).compute().data*365
        spring = z_gb.sel(season='MAM')[f'{var}'].mean(dim=['x','y']).compute().data*365
        summer = z_gb.sel(season='JJA')[f'{var}'].mean(dim=['x','y']).compute().data*365
        fall = z_gb.sel(season='SON')[f'{var}'].mean(dim=['x','y']).compute().data*365
        
    d = {'RGIId':rgi_id, 'var': var, 'z':z, 'winter': winter,
             'spring':spring, 'summer': summer, 'fall':fall}
            
    df = pd.DataFrame(d, index=[0])
    
    return df
        
    