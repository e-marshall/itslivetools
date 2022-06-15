import geopandas as gpd
import os
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from shapely.geometry import Polygon
from shapely.geometry import Point
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy
import cartopy.feature as cfeature
import json

def read_in_s3(http_url):
    s3_url = http_url.replace('http','s3')
    s3_url = s3_url.replace('.s3.amazonaws.com','')

    datacube = xr.open_dataset(s3_url, engine = 'zarr',
                                storage_options={'anon':True},
                                chunks = 'auto')

    return datacube

def find_granules_by_zone(input_dict, epsg_code):
    '''This function takes a dictionary (itslive catalog geojson) and a epsg code referencing region of interest. 
    returns list of urls corresponding to datacubes stored in s3 buckets where links *exist*'''
    fs = s3fs.S3FileSystem(anon=True)

    url_ls = []
    for granule in range(len(input_dict['features'])):
        if input_dict['features'][granule]['properties']['data_epsg'] == epsg_code:

            #format question - better to condense this into 1 line or break into 3 to be more readable?
            http_url = input_dict['features'][granule]['properties']['zarr_url']
            s3_url = http_url.replace('http','s3').replace('.s3.amazonaws.com','')

            if fs.lexists(s3_url) == True:
                url_ls.append(s3_url)

    return url_ls

def find_granule_by_point(input_dict, input_point): #[lon,lat]
    '''Takes an inputu dictionary (a geojson catalog) and a point to represent AOI.
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



def get_bbox_single(input_xr):
    
    '''Takes input xr object (from itslive data cube), plots a quick map of the footprint. 
    currently only working for granules in crs epsg 32645'''

    xmin = input_xr.coords['x'].data.min()
    xmax = input_xr.coords['x'].data.max()

    ymin = input_xr.coords['y'].data.min()
    ymax = input_xr.coords['y'].data.max()

    pts_ls = [(xmin, ymin), (xmax, ymin),(xmax, ymax), (xmin, ymax), (xmin, ymin)]

    #print(input_xr.mapping.spatial_epsg)
    #print(f"epsg:{input_xr.mapping.spatial_epsg}")
    crs = f"epsg:{input_xr.mapping.spatial_epsg}"
    #crs = {'init':f'epsg:{input_xr.mapping.spatial_epsg}'}
    #crs = 'epsg:32645'
    #print(crs)

    polygon_geom = Polygon(pts_ls)
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom]) 
    polygon = polygon.to_crs('epsg:4326')

    bounds = polygon.total_bounds
    bounds_format = [bounds[0]-15, bounds[2]+15, bounds[1]-15, bounds[3]+15]

    states_provinces = cfeature.NaturalEarthFeature(
        category = 'cultural',
        name = 'admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none'
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(states_provinces)

    ax.set_extent(bounds_format, crs = ccrs.PlateCarree())

    polygon.plot(ax=ax, facecolor = 'none', edgecolor='red', lw=1.)

    return polygon

def get_bbox_group(input_ls, bounds = [-180, 180, -90, 90]): 
    
    '''plots the spatial extents of a list of datacubes'''
    
    poly_ls = []
    
    for xr_obj in range(len(input_ls)):
        '''Takes input xr object (from itslive data cube), plots a quick map of the footprint. 
        currently only working for granules in crs epsg 32645'''

        xmin = input_ls[xr_obj].coords['x'].data.min()
        xmax = input_ls[xr_obj].coords['x'].data.max()
        ymin = input_ls[xr_obj].coords['y'].data.min()
        ymax = input_ls[xr_obj].coords['y'].data.max()

        pts_ls = [(xmin, ymin), (xmax, ymin),(xmax, ymax), (xmin, ymax), (xmin, ymin)]

        
        crs = f"epsg:{input_ls[xr_obj].mapping.spatial_epsg}" #should be format: 'epsg:32645'
        #print(crs)

        polygon_geom = Polygon(pts_ls)
        polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom]) 
        polygon = polygon.to_crs('epsg:4326')
        poly_ls.append(polygon)

    bounds_format = [bounds[0], bounds[1], bounds[2], bounds[3]]
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = ccrs.PlateCarree())
    ax.stock_img()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent(bounds_format, crs = ccrs.PlateCarree())
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = False
    gl.xlines = False
    gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'color': 'black'}

    for element in range(len(poly_ls)):
            
        #polygon.plot(ax=ax, facecolor = 'none', edgecolor='red', lw=1.)
        poly_ls[element].plot(ax=ax, facecolor='none', edgecolor='red', lw=1.)
