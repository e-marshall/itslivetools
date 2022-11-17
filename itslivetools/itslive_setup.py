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