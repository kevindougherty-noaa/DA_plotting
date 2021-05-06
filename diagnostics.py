import numpy as np
import pandas as pd
import xarray as xr
from xhistogram.xarray import histogram
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interpn
from matplotlib.colors import Normalize
from matplotlib import rcParams, ticker, cm

import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def _get_linear_regression(data1, data2):
    """
    Inputs:
        data1 : data on the x axis
        data2 : data on the y axis
    """
    x = np.array(data1).reshape((-1, 1))
    y = np.array(data2)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    intercept = model.intercept_
    slope = model.coef_[0]
    # This is the same as if you calculated y_pred
    # by y_pred = slope * x + intercept
    y_pred = model.predict(x)
    return y_pred, r_sq, intercept, slope

def bin_df(df, variable, dlat=5, dlon=5, return_count=False):
    """
    Bin a variable from a datafram given dlat, dlon using xhistogram
    Input:
        df: dataframe that needs to binned
        variable: column from dataframe that needs to be binned
        dlat: latitude bin in degrees (Default: 5)
        dlon: longitude bin in degrees (Default: 5)
    Return:
        bindata: DataArray of binned data
    """
    bindata = df.reset_index(level=[0]).to_xarray()

    lon_bins = np.arange(0, 360+dlon, dlon)
    lat_bins = np.arange(-90, 90+dlat, dlat)

    # chunk data
    bindata_chunked = bindata.chunk({'index': '5MB'})

    count = histogram(bindata_chunked['latitude@MetaData'],
                        bindata_chunked['longitude@MetaData'],
                        bins=[lat_bins, lon_bins])

    # create bins with count
    bindata = histogram(bindata_chunked['latitude@MetaData'],
                        bindata_chunked['longitude@MetaData'],
                        bins=[lat_bins, lon_bins],
                        weights = bindata_chunked[variable])

    bindata = bindata / count.where(count > 1)

    if return_count:
        return bindata, count
    else:
        return bindata

def spatial_binned(binned_data, metadata):
    """
    Plot a global map of data that was binned using bin_df()
    Input:
        binned_data: DataArray of binned data
        metadata: dictionary of metadata
    Output:
        Saved .png file
    """
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax.add_feature(cfeature.GSHHSFeature(scale='auto'))
    ax.set_extent([-180, 180, -90, 90])
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    cs = plt.pcolormesh(binned_data['longitude@MetaData_bin'], binned_data['latitude@MetaData_bin'],
                        binned_data.values, cmap=metadata['cmap'],
                        vmin=metadata['vmin'], vmax=metadata['vmax'],
                        transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs, shrink=0.5, pad=.03, extend='both')
    cb.set_label(metadata['label'])
    plt.title(metadata['title'], loc='left')
    plt.title(metadata['cycle'], loc='right',fontweight='semibold')

    plt.savefig(metadata['outfig'], bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def spatial(df, metadata, variable=None):
    """
    Plot a global spatial map of data
    Input:
        df: dataframe with latitude/longitude data included
        variable: column from dataframe to be plotted (Default: None
                  plots spatial coverage)
        metadata: dictionary of metadata
    Output:
        Saved .png file
    """
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    ax.add_feature(cfeature.GSHHSFeature(scale='auto'))
    ax.set_extent([-180, 180, -90, 90])
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    if variable is not None:
        cs = plt.scatter(df['longitude@MetaData'], df['latitude@MetaData'], c=df[variable], s=10,
                    cmap=metadata['cmap'], transform=ccrs.PlateCarree())
        cb = plt.colorbar(cs, shrink=0.5, pad=.03, extend='both')
        cb.set_label(metadata['label'])
        
    else:
        cs = plt.scatter(df['longitude@MetaData'], df['latitude@MetaData'], s=10,
                    color=metadata['cmap'], transform=ccrs.PlateCarree())
        
    
    plt.title(metadata['title'], loc='left')
    plt.title(metadata['cycle'], loc='right',fontweight='semibold')
    
    plt.savefig(metadata['outfig'], bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

def scatter(dfX, dfY, metadata, density=False):
    # generate and save scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    y_pred, r_sq, intercept, slope = _get_linear_regression(dfX, dfY)
    if density:
        density_scatter(dfX.values, dfY.values, ax=ax, fig=fig,
                        bins=[100, 100], s=4, cmap='magma')
    else:
        plt.scatter(x=dfX, y =dfY, s=4, color='darkgray', label=f'n={dfX.count()}')
    label = f'y = {slope:.4f}x + {intercept:.4f}\nR\u00b2 : {r_sq:.4f}'
    plt.plot(dfX, y_pred, color='black', linewidth=1, label=label)
    plt.legend(loc='upper left', fontsize=11)
    plt.title(metadata['title'], loc='left')
    plt.xlabel(metadata['xlabel'], fontsize=12)
    plt.ylabel(metadata['ylabel'], fontsize=12)
    plt.title(metadata['cycle'], loc='right', fontweight='semibold')
    plt.savefig(metadata['outfig'], bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    
    return None

def lineplot(dfX, dfY, metadata):
    # generate and save line plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.plot(dfX, dfY, linestyle=metadata['linestyle'], linewidth=metadata['linewidth'],
             color=metadata['color'], label=metadata['label'])
    plt.grid()
    plt.legend(loc='upper left', fontsize=11)
    plt.title(metadata['title'], loc='left')
    plt.xlabel(metadata['xlabel'], fontsize=12)
    plt.ylabel(metadata['ylabel'], fontsize=12)
    plt.title(metadata['cycle'], loc='right', fontweight='semibold')
    plt.savefig(metadata['outfig'], bbox_inches='tight', pad_inches=0.1)
    plt.close('all')

    return None

def density_scatter(x, y, ax=None, fig=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])),
                data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)
    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    ax.scatter(x, y, c=z, **kwargs)
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    #cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')
    return ax