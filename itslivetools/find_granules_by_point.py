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