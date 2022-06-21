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
