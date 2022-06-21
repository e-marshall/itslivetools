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
