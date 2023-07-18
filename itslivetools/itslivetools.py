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
def get_bbox(input_xr, epsg=None):
    
    '''Takes input xr object (from itslive data cube), plots a quick map of the footprint. 
    currently only working for granules in crs epsg 32645'''

    xmin = input_xr.coords['x'].data.min()
    xmax = input_xr.coords['x'].data.max()

    ymin = input_xr.coords['y'].data.min()
    ymax = input_xr.coords['y'].data.max()

    pts_ls = [(xmin, ymin), (xmax, ymin),(xmax, ymax), (xmin, ymax), (xmin, ymin)]

   
    crs = f"epsg:{epsg}"
    print(crs)
    
    polygon_geom = Polygon(pts_ls)
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom]) 
    #polygon = polygon.to_crs('epsg:4326')

    return polygon


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

        
 class IndGlacier:
    
    def __init__(self, rgi_id, rgi_outline, dem, itslive, centerLine=None, ablationLine = None, lowestPoint = None):
        
        self.rgi_id = rgi_id
        self.rgi_outline = rgi_outline
        self.dem = dem
        self.itslive = itslive
        self.centerLine = centerLine
        self.ablationLine = ablationLine
        self.lowestPoint = lowestPoint
    
    def calc_flowline(self):
        '''function to calculate distance from glacier terminus along flowline. should ultimately be a method of the Glacier class.
    takes an endpoint object and a ablation line vector. should return a 1D array with distance from terminus (or maybe a geodataframe with each point and dist as an attr??
    NO - doesn't take endpont, only ablation line because the end point does not always lie along the ablation line -- not sure the best way to handle this'''

   
    
        dl = np.zeros(len(list(self.ablationLine['geometry'].iloc[0].coords))) # make an empty array the list of the ablation line vector

        #dl = np.zeros(5)

        coords = list(self.ablationLine['geometry'].iloc[0].coords)

        coords_X = [coords[point][0] for point in range(len(coords))]
        coords_Y = [coords[point][1] for point in range(len(coords))]

        dl[1:] = np.sqrt(np.diff(coords_X)**2 + np.diff(coords_Y)**2)

        dist_profile = pd.DataFrame({'x_coords': coords_X,
                                 'y_coords': coords_Y,
                                 'distance': np.cumsum(dl)})
    
        geometry = gpd.points_from_xy(dist_profile['x_coords'], dist_profile['y_coords'])

        gdf = gpd.GeoDataFrame(
            dist_profile, geometry= geometry)
        
        gdf['ind_dist'] = gdf['distance'].diff()
        gdf['x_diff'] = gdf['x_coords'].diff()
        gdf['y_diff'] = gdf['y_coords'].diff()

        # 'downsample' remove points closer to one another than 50 meters
        #gdf_downsamp = gdf.loc[gdf['ind_dist'] >= 30]
        # ^^^^ should I downsample or not? turns out that wasn't the memory issue so don't need to

        #make list of tuples of flowline coords
        flowline_points_ls = [(gdf['x_coords'].iloc[row], gdf['y_coords'].iloc[row]) for row in range(len(gdf))]
        #list into array
        flowline_points = np.array(flowline_points_ls)
        #make xr objects of x and y coordinates with a new dimension 'points'
        x = xr.DataArray(flowline_points[:,0], dims = 'points')
        y = xr.DataArray(flowline_points[:,1], dims = 'points')
        #make new xr object that is velocity data interpolated onto the points
        interp_points = self.itslive.interp(x=x, y=y)
        #reverse index so that first points are terminus, last are end of ablation zone
        interp_points = interp_points.reindex(points = list(reversed(interp_points.points)))
        
        flowline_xr = gdf.to_xarray()
        flowline_xr = flowline_xr.reindex(index = list(reversed(flowline_xr.index)))

        interp_points['distance'] = ('points', flowline_xr.distance.data )



        return interp_points
    

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
