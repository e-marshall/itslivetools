# tools for retreat stac catalog build for expanding retreat proejct (at some point need to merge w/ new_retreat
from pathlib import Path
import stackstac
import xarray as xr
import glob
import shutil
import pystac
from itertools import groupby
from shapely.geometry import Polygon, mapping
import rasterio as rio
import os
import pandas as pd
from rasterio.warp import transform_bounds, transform_geom
from pystac.extensions import item_assets 
from pystac.extensions.projection import AssetProjectionExtension



def move_files_to_tile_subdir(path, tile):
    
    for data in glob.glob(f'{path}{tile}*'):
        shutil.move(data, f'{path}/{tile}')
#https://stackoverflow.com/questions/28913088/moving-files-with-wildcards-in-python   

def data_directory_setup(path):
    
    folder = Path(path)
    files_ls = os.listdir(folder)
    #make list of all files in dir
    files_ls = [x for x in files_ls if 'filelist_velocities' not in x]
    #find unique footprints in the files list
    footprint_ids = set([x[:10] for x in files_ls])
    print(len(footprint_ids))
    
    #make subdirectories for each footprint
    for tile in footprint_ids:
        
        full_path = os.path.join(path, tile)
        os.makedirs(full_path, exist_ok=True)
        move_files_to_tile_subdir(path, tile)
        #for data in glob.glob(f'{path}{tile}*'):
        #shutil.move(data, f'{path}/{tile}')
        
    

def parse_fname(fposix):
    fname = fposix.stem
    prevar = fname.split('-')[0]
    var = fname.split('-')[1].split('+')[0]
    post_var = fname.split('-')[1].split('+')[1]
    
    sensor = post_var[:2]
    #acq_date = pd.to_datetime(post_var[3:18])
    temp = post_var.split('S1_')
    img1_date = temp[1].split('_')[0]
    img1_id = temp[1].split('_')[1]
    
    #acq_id = post_var[19:23]
    
    #sec_date = pd.to_datetime(post_var[39:54])
    img2_date = temp[2].split('_')[0]
    img2_id = temp[2].split('_')[1]
    #sec_id = post_var[55:59]
    
    #print('sensor: ', sensor)
    #print('acq_date: ', acq_date)
    #print('acq_id: ', acq_id)
    #print('sec date: ', sec_date)
    #print('sec id : ', sec_id)
    
    
    site = prevar[:2]
    frame = prevar[3:6]
    orbit = prevar[7:10]
    mid_date = pd.to_datetime(prevar[11:19])
    #print('site: ', site)
    #print('frame: ', frame)
    #print('orbit: ', orbit)
    #print('mid_date: ', mid_date)
    
    return sensor, img1_date, img1_id, img2_date, img2_id, site, orbit, frame, mid_date

#fn from TDS article
def get_bbox_and_footprint(dataset):
    #create boundingbox, will depent on if it comes from rasterio or rioxarray
    
    bounds = dataset.bounds
    
    if isinstance(bounds, rio.coords.BoundingBox):
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
    else:
        bbox = [float(f) for f in bounds()]
    
    #create footprint
    footprint = Polygon([
        [bbox[0], bbox[1]],
        [bbox[0], bbox[3]],
        [bbox[2], bbox[3]],
        [bbox[2], bbox[1]]
    ])
    return bbox, mapping(footprint)

    return bbox, mapping(footprint)

def build_stac_catalog_single_var(catalog, dis_angs, var_name):
    
    for dis_ang in dis_angs:

            if str(dis_ang).endswith('.tif'):

                try:
                    sensor, img1_date, img1_id, img2_date, img2_id, site, orbit, frame, mid_date = parse_fname(dis_ang)
                except: 
                    pass
                #open file with rasterio
                ds = xr.open_dataset(dis_ang, engine='rasterio').squeeze()   #bounds should be: left bottom right top

                left = ds.x.data.min()
                right = ds.x.data.max()
                bottom = ds.y.data.min()
                top = ds.y.data.max()

                #create bbox, footprint
                bbox = [left, bottom, right, top]
                footprint = Polygon([
                    [bbox[0], bbox[1]],
                    [bbox[0], bbox[3]],
                    [bbox[2], bbox[3]],
                    [bbox[2], bbox[1]]
                    ])
                footprint_m = mapping(footprint)


                #project to wgs84, optain in geometric coords
                geo_bounds = transform_bounds(ds.rio.crs, 'EPSG:4326', *bbox)
                geo_footprint = transform_geom(ds.rio.crs, 'EPSG:4326', footprint_m)

                #properties
                idx = dis_ang.stem[:19]
                #date = pd.to_datetime(dis_ang.stem[11:19])
                date = pd.to_datetime(dis_ang.stem[11:19])
                tile = dis_ang.stem[:10]

                item = pystac.Item(
                    id = idx,
                    geometry = geo_footprint,
                    bbox = geo_bounds,
                    datetime = date,
                    stac_extensions = ['https://stac-extensions.github.io/projection/v1.0.0/schema.json'],
                    properties = dict(
                        tile=tile,
                        sensor = sensor,
                        img1_date = img1_date,
                        img1_id = img1_id,
                        img2_date = img2_date,
                        img2_id = img2_id,
                        site = site, 
                        orbit = orbit, 
                        frame = frame, 
                    )
            )

                catalog.add_item(item)
                idx = dis_ang.stem[:19]
                item = catalog.get_item(idx)

                #as before, open both with rasterio to get bbox, footprint
                ds = rio.open(dis_ang)
                bbox, footprint = get_bbox_and_footprint(ds)
                #print(dis_ang.as_posix())
                item.add_asset(
                    key = var_name,
                  #  title='dis_ang',
                    asset = pystac.Asset(
                        href = dis_ang.as_posix(),
                        title=var_name,
                        media_type = pystac.MediaType.GEOTIFF
                    )
                )
                #extend the asset wtih extension projection
                asset_ext = AssetProjectionExtension.ext(item.assets[var_name])
                asset_ext.epsg = ds.crs.to_epsg()
                asset_ext.shape = ds.shape
                asset_ext.bbox = bbox
                asset_ext.geometry = footprint
                asset_ext.transform = [float(getattr(ds.transform, letter)) for letter in 'abcdef']
            else:
                pass
    print(len(list(catalog.get_items())))


def add_asset_to_stac_catalog(catalog, var_ls, var_name):
    
    for file in var_ls:
        
        if str(file).endswith('.tif'):
            
            idx = file.stem[:19]
            item = catalog.get_item(idx)
            item.extra_fields[f'orig_file_{var_name}'] = str(file)[42:]
            
            ds = rio.open(file)
            bbox, footprint = get_bbox_and_footprint(ds)
            
            item.add_asset(
                key=var_name,
                asset = pystac.Asset(
                    href = file.as_posix(),
                    title=var_name,
                    media_type = pystac.MediaType.GEOTIFF
                )
            )
            asset_ext = AssetProjectionExtension.ext(item.assets[var_name])
            asset_ext.epsg = ds.crs.to_epsg()
            asset_ext.shape = ds.shape
            asset_ext.bbox = bbox
            asset_ext.geometry = footprint
            asset_ext.transform = [float(getattr(ds.transform, letter)) for letter in 'abcdef']
        else:
            pass
    print(len(list(catalog.get_items())))