import geopandas as gpd
import os
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from shapely.geometry import Polygon
from shapely.geometry import Point
import json
from scipy.stats import sem
import scipy
from statsmodels.stats.stattools import medcouple
import math


def find_granule_by_point(input_dict, input_point): #[lon,lat]
    '''Takes an input dictionary (a geojson catalog) and a point to represent AOI.
    this returns a list of the s3 urls corresponding to zarr datacubes whose footprint covers the AOI'''
    
    target_granule_urls = []
   
    point_geom = Point(input_point[0], input_point[1])
    point_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry = [point_geom])
    for granule in range(len(input_dict['features'])):
        
        bbox_ls = input_dict['features'][granule]['geometry']['coordinates'][0]
        bbox_geom = Polygon(bbox_ls)
        bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry = [bbox_geom])
        
        if bbox_gdf.contains(point_gdf).all() == True:
            target_granule_urls.append(input_dict['features'][granule]['properties']['zarr_url'])
        else:
            pass
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

def velocity_clip(ds, rgi):
    
    ds_clip = ds.rio.clip(rgi.geometry, rgi.crs)
    
    return ds_clip

def dem_clip(ds, dem, rgi):
    
    dem_clip = dem.rio.clip(rgi.geometry, rgi.crs)
    ds['z'] = dem_clip.Band1
    
    return ds

def elevation_format(ds):
    
    z_lower = ds.where(ds.z <= np.nanmedian(ds.z.data), drop=True)
    z_upper = ds.where(ds.z > np.nanmedian(ds.z.data), drop=True)
    
    ds['z_lower'] = z_lower.z
    ds['z_upper'] = z_upper.z
    
    return ds

def calc_valid(arr):
    
    i = np.count_nonzero(~np.isnan(arr))
    
    return i

def calc_full_cov(ds):
    
    possible = np.count_nonzero(~np.isnan(ds.v.mean(dim='mid_date')))
    
    actual = xr.apply_ufunc(calc_valid,
                            ds.v,
                            input_core_dims=[['x','y']],
                            exclude_dims=set(('x','y')),
                            vectorize=True,
                            dask='parallelized')
    cov = actual / possible
    ds['full_actual'] = actual
    ds['full_possible'] = possible
    ds['full_cov'] = cov
    
    return ds

def calc_lower_cov(ds):
    
    ds_sub = ds.where(ds.z <= np.nanmedian(ds.z.data),drop=True)
    possible = np.count_nonzero(~np.isnan(ds_sub.v.mean(dim='mid_date')))
    actual = xr.apply_ufunc(calc_valid,
                            ds_sub.v,
                            input_core_dims=[['x','y']],
                            exclude_dims=set(('x','y')),
                            vectorize=True,
                            dask='parallelized')
    cov = actual / possible
    ds['lower_actual'] = actual
    ds['lower_possible'] = possible
    ds['lower_cov'] = cov
    return ds

def calc_upper_cov(ds):
    
    ds_sub = ds.where(ds.z > np.nanmedian(ds.z.data),drop=True)
    possible = np.count_nonzero(~np.isnan(ds_sub.v.mean(dim='mid_date')))
    actual = xr.apply_ufunc(calc_valid,
                            ds_sub.v,
                            input_core_dims=[['x','y']],
                            exclude_dims=set(('x','y')),
                            vectorize=True,
                            dask='parallelized')
    
    cov = actual / possible
    ds['upper_actual'] = actual
    ds['upper_possible'] = possible
    ds['upper_cov'] = cov
    return ds

def calc_area_cov_index(ds):
    
    aci = (ds['upper_cov'] / ds['full_cov']) * (ds['upper_cov'] / (ds['upper_cov'] + ds['lower_cov']))
    
    ds['aci'] = aci
    
    return ds

def add_mask_var(ds, rgi):
    '''rasterize shapefile of single glacier and add it as a var to ds object
    '''
    outline_mask = make_geocube(
        vector_data = rgi,
        measurements=['Area'],
        like=ds.v,
        fill=-999.)
    ds['mask'] = outline_mask['Area']
    
    return ds

def trim_cov(ds):
    
    #ds = ds.where(ds['full_cov'] >= threshold, drop=True)
    ds = ds.where(ds['full_cov'] != 0,drop=True)
    
    return ds 

def itslive_processing_driver(itslive_raster, dem_raster, rgi_id, rgi_gpdf):
    
    rgi_single = rgi_gpdf.loc[rgi_gpdf['RGIId'] == rgi_id]
    
    ds_clip = velocity_clip(itslive_raster, rgi_single)
    print('velocity cilpped')
    ds_clip = add_mask_var(ds_clip, rgi_single)
    
    ds_clip = dem_clip(ds_clip, dem_raster, rgi_single)
    print('dem clipped')
    ds_clip = elevation_format(ds_clip)
    
    ds_clip = ds_clip.load()
    print('obj loaded')
    ds_clip = calc_full_cov(ds_clip)
    
    ds_clip = calc_lower_cov(ds_clip)
    
    ds_clip = calc_upper_cov(ds_clip)
    
    ds_clip = calc_area_cov_index(ds_clip)
    print('cov stuff done')
    ds_clip = trim_cov(ds_clip)
    print('cov trim done')
    #ds_clip = reformat_attrs(ds_clip)
    #print('attrs reformatted')
    return ds_clip





def calc_seasonal_sem_by_z(input_gb, z, var, rgi_id):
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
        
        winter = z_gb.sel(season='DJF')[f'{var}'].mean(dim=['x','y']).compute().data
        spring = z_gb.sel(season='MAM')[f'{var}'].mean(dim=['x','y']).compute().data
        summer = z_gb.sel(season='JJA')[f'{var}'].mean(dim=['x','y']).compute().data
        fall = z_gb.sel(season='SON')[f'{var}'].mean(dim=['x','y']).compute().data
        
    d = {'RGIId':rgi_id, 'var': var, 'z':z, 'winter': winter,
             'spring':spring, 'summer': summer, 'fall':fall}
            
    df = pd.DataFrame(d, index=[0])
    
    return df
        
def wrapper_all_glaciers(xr_dict):
    
    df_v_ls, df_sem_ls = [],[]
    
    for key in xr_dict.keys():
    
        df_v = pd.concat([calc_seasonal_mean_v_by_z(xr_dict[key], z, 'v', key) for z in ['z0','z1','z2','z3','full']])
        df_sem = pd.concat([calc_seasonal_sem_by_z(xr_dict[key], z, 'sem_v', key) for z in ['z0','z1','z2','z3','full']])
        df_v_ls.append(df_v)
        df_sem_ls.append(df_sem)
    
    df_full = pd.concat([df_v_ls, df_sem_ls])
    return df_full

def itslive_dict_process(itslive_dict):
    '''changing object types of some variables to save to nc. this is for saving a dict of xr objects to file. the xr objects are itslive velocities clipped to outline of ind glaciers. full outline not centerline'''
    for key in itslive_dict.keys():
        
        itslive_dict[key]['autoRIFT_software_version'] = itslive_dict[key]['autoRIFT_software_version'].astype(str)
        itslive_dict[key]['granule_url'] = itslive_dict[key]['granule_url'].astype(str)
        itslive_dict[key]['mission_img1'] = itslive_dict[key]['mission_img1'].astype(str)
        itslive_dict[key]['mission_img2'] = itslive_dict[key]['mission_img2'].astype(str)
        itslive_dict[key]['satellite_img1'] = itslive_dict[key]['satellite_img1'].astype(str)
        itslive_dict[key]['satellite_img1'] = itslive_dict[key]['satellite_img1'].astype(str)
        itslive_dict[key]['satellite_img2'] = itslive_dict[key]['satellite_img2'].astype(str)
        itslive_dict[key]['satellite_img2'] = itslive_dict[key]['satellite_img2'].astype(str)
        itslive_dict[key]['sensor_img1'] = itslive_dict[key]['sensor_img1'].astype(str)
        itslive_dict[key]['sensor_img2'] = itslive_dict[key]['sensor_img2'].astype(str)
        
    return itslive_dict

## EXploratory analysis tools
def timespan_plot(ds, color,alpha):
    
    ax.hlines(xmin=ds.acquisition_date_img1, xmax=ds.acquisition_date_img2, y=ds.v.mean(dim=['x','y']), color=color, alpha = alpha)

def add_time_separation(ds):
    
    ds['img_separation'] = (ds.acquisition_date_img1 - ds.acquisition_date_img2).astype('timedelta64[D]') / np.timedelta64(1,'D')*-1
    return ds

def trim_img_separation(ds):
    ds = ds.sortby('mid_date', ascending=True)
    ds_short = ds.where(ds.img_separation <= 96, drop=True)
    return ds_short

def mc(arr, axis='xy'):
    #print(f'arr: {arr.shape}')
    
    arr = np.reshape(arr,(len(arr)))
    #print(arr.shape)
    s = pd.Series(arr).dropna()
    #print(s.shape)
    mc = medcouple(s)
    #print(mc)
    #print('----')
    
    return mc

def add_MC_var(ds):
    
    ds_vxy = ds_short.v.stack(xy=('x','y'))
    
    res_mc = xr.apply_ufunc(mc,
                     ds_vxy,
                     input_core_dims=[['xy'],],
                     #exclude_dims=set(('xy',)),
                     #output_core_dims=[['xy']],
                     vectorize=True,
                    )
    ds['MC_v'] = res_mc
    
    return ds


def adj_boxplot(v):
    '''function to calculate adjusted boxplot for skewed distributes (from Hubert + Vanderviernan 2008) 
    for outlier detection of velocity data
    '''
   
    v_stack = v.flatten()
    
    q1 = np.nanpercentile(v_stack, 25)
    q3 = np.nanpercentile(v_stack, 75) 
    iqr = (q3 - q1)
    
    #arr = np.reshape(v_stack,(len(v_stack)))
    #print(arr.shape)
    s = pd.Series(v_stack).dropna()
    mc = medcouple(s)
    
    iqr = (q3-q1)
    if mc >= 0:
        lb = q1 - (1.5*math.exp(-4*mc)*iqr)
        ub = q3 + (1.5*math.exp(3*mc)*iqr)
    elif mc < 0:
        lb = q1 - (1.5*math.exp(-3*mc)*iqr)
        ub = q3 + (1.5*math.exp(4*mc)*iqr)
    
    filtered = np.where(np.logical_and((lb <= v),(v <= ub)), v, np.nan)
   
    return filtered

def outlier_detection_adj_boxplot(ds): #v variable hardcoded
    '''
    function to apply the adjusted boxplot function as a vectorized function along every element of the mid_date dim
    '''
       
    v_filtered = xr.apply_ufunc(adj_boxplot, #function you want to broadcast
                        ds.v,                #input to the function
                        input_core_dims = [['x','y']], #shape of the above object
                        output_core_dims =[['x','y']], # shape of the output object (what are you doing to the data - transforming, reducing? 
                        exclude_dims = set(('x','y')), # what dimensions won't be changed -ie if you're reducing, x and y will change. if you're turning pixels to nans, no dims change
                        vectorize=True, ) 
    ds['v_filtered'] = v_filtered
    
    return ds

#def filtering_wrapper_w_percentile_subset(itslive_dict):
    

def read_and_process_to_dict(path_to_files_dir):
    '''takes a path to a directory of .nc files (processed in XXXXXX)
    for every file in dir:
    - reads in as xr object
    - adds time separation var
    - sorts by middate
    - subsets to img separation < 96 days
    - applies MC outlier detection filter
    - calculates 98th percentile of magnitude of velocity var for each timestep
    - adds 98th percentile velocity var
    - subsets to only the timesteps where date1, date2 in the same season
    returns all as dictionary where key: rgiid, val:xr object
    '''
    
    files = os.listdir(path_to_files_dir)
    files = [f for f in files if 'rgi' not in f]
    #print(files[0])
    key_ls, val_ls = [],[]
    
    for f in range(len(files)):
        #print(files[f])
        
        glacier = files[f][3:-3]
        print(glacier)
        key_ls.append(glacier)
        ds = xr.open_dataset(os.path.join(path_to_files_dir, files[f]), engine='netcdf4')
        ds = add_time_separation(ds)
        ds = ds.sortby('mid_date', ascending=True)
        ds_short = ds.where(ds.img_separation <= 96., drop=True)
        ds_filtered = outlier_detection_adj_boxplot(ds_short)
        vf_98 = np.nanpercentile(ds_filtered.v_filtered.stack(xyt = ('x','y','mid_date')), 98)
        vf_98_sub = ds_filtered.where(ds_filtered.v_filtered <= vf_98, drop=True)
        ds_filtered['vf_98'] = vf_98_sub.v_filtered
        ds_filtered = ds_filtered.where(ds_filtered.acquisition_date_img1.dt.season == ds_filtered.acquisition_date_img2.dt.season, drop=True)
        
        ds_filtered['vf_98_mday'] = ds_filtered['vf_98'] / 365 
    
        val_ls.append(ds_filtered)
    
    glacier_dict = dict(zip(key_ls, val_ls))
    
    return glacier_dict
        