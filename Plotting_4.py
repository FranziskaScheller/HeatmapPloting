import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
#import warnings
#warnings.filterwarnings(action=”ignore”)

def create_geodf_from_df(df: pd.DataFrame, lon_name: str = 'longitude', lat_name: str = 'latitude') -> geopandas.GeoDataFrame:
    """ Create Geopandas.GeoDataFrame from df using the longitudinal and latitudinal information in columns lon_name and lat_name
    :param df: DataFrame with geographical information in columns lon_name and lat_name
    :param lon_name: column name with longitudinal information (default 'lon')
    :param lat_name: column name with latitudinal information (default 'lat')
    :return: geopandas.GeoDataFrame including columns of df
    """
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(x=df[lon_name], y=df[lat_name]))

# def scatter_points_germany(gdf: geopandas.GeoDataFrame,
#                            col_to_plot: str,
#                            ax: [plt.Axes, None],
#                            plot_colorbar: bool = True,
#                            **kwds_plot) -> plt.Axes:
#     """ Create scatter plot of values in gdf[col_to_plot] over boundary of germany.
#     :param gdf: geopandas data frame
#     :param col_to_plot: the column of gdf to plot
#     :param ax: (optional) a plt.Axes object where to plot
#     :param plot_colorbar: whether to plot a colorbar besides the plot
#     :param kwds_plot: passed to geopandas scatter function
#     :return: plt.Axes object with plot
#     """
#     world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#     germany = world[world['name'] == 'Germany']
#     if ax is None:
#         fig, ax = plt.subplots()
#     germany.boundary.plot(edgecolor='black', ax=ax)
#     ax.set(xlabel='longitude', ylabel='latitude')
#     if plot_colorbar:
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', legend=True, cax=cax, **kwds_plot)
#     else:
#         gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
#     return ax

def scatter_points_europe(gdf: geopandas.GeoDataFrame,
                           col_to_plot: str,
                           ax: [plt.Axes, None],
                           plot_colorbar: bool = True,
                           **kwds_plot) -> plt.Axes:
    """ Create scatter plot of values in gdf[col_to_plot] over boundary of germany.
    :param gdf: geopandas data frame
    :param col_to_plot: the column of gdf to plot
    :param ax: (optional) a plt.Axes object where to plot
    :param plot_colorbar: whether to plot a colorbar besides the plot
    :param kwds_plot: passed to geopandas scatter function
    :return: plt.Axes object with plot
    """
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    europe = world[world.continent == "Europe"]
    #europe = world[(world.continent == "Europe") & (world.name != "Russia")]

    # Create a custom polygon
    polygon = Polygon([(-22, 26.5), (45.5, 26.5), (45.5, 72.5), (-22, 72.5)])
    # poly_gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)
    # fig, ax = plt.subplots()
    # ax = europe.plot(ax=ax)
    # poly_gdf.plot(edgecolor=”red”, ax = ax, alpha = 0.1)
    # plt.show()

    europe = geopandas.clip(europe, polygon)
    europe.plot()

    if ax is None:
        fig, ax = plt.subplots()
    europe.boundary.plot(edgecolor='black', ax=ax)
    ax.set(xlabel='longitude', ylabel='latitude')
    #if plot_colorbar:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    #aspect='1.3',
    gdf.plot(ax=ax, marker='.', column=col_to_plot, legend=True, cax=cax, **kwds_plot)
    #ax.set(title='test')
    # else:
    #     gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)
    #gdf.plot(ax=ax, marker='.', column=col_to_plot, alpha=.1, aspect='1.3', **kwds_plot)

    return ax


DF_Data = pd.read_csv(
    'MeanClimatologicalData_DF_neighboringcountries_reshaped.csv', header=0, index_col=None, sep=';')
DF_Data = create_geodf_from_df(DF_Data)

#world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#europe=world[world.continent==”Europe”]

# for season in seasons:
#     wind_speeds, lat, lon, time = read_nc4_file(file=nc4_file)
#     df_mean_std = pd.DataFrame({
#         'lat': lat,
#         'lon': lon,
#         'mean': np.mean(wind_speeds, axis=0),
#         'std': np.sqrt(np.var(wind_speeds, axis=0))
#     })
#     gdf_mean_std = create_geodf_from_df(df_mean_std)
#gdf_loadings = create_geodf_from_df(DF_Data)

# Air pressure poland
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'msl_PL', ax=ax, cmap='magma')
ax.set_title('Mean air pressure at mean sea level during a Dunkelflaute [hPa]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('MSLP_PL.png')


# Air pressure netherlands
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'msl_NL', ax=ax, cmap='magma')
ax.set_title('Mean air pressure at mean sea level during a Dunkelflaute [hPa]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('MSLP_NL.png')

# Air pressure netherlands
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'msl_FR', ax=ax, cmap='magma')
ax.set_title('Mean air pressure at mean sea level during a Dunkelflaute [hPa]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('MSLP_FR.png')

# Temperature poland
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 't2m_PL', ax=ax, cmap='magma')
ax.set_title('Mean temperature at 2 meters during a Dunkelflaute [$^\circ$C]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('T2M_PL.png')


# Temperature netherlands
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 't2m_NL', ax=ax, cmap='magma')
ax.set_title('Mean temperature at 2 meters during a Dunkelflaute [$^\circ$C]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('T2M_NL.png')

# Temperature france
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 't2m_FR', ax=ax, cmap='magma')
ax.set_title('Mean temperature at 2 meters during a Dunkelflaute [$^\circ$C]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('T2M_FR.png')

# Solar poland
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'ssrdPL', ax=ax, cmap='magma')
ax.set_title('Mean solar radiation at the surface during a Dunkelflaute [W m $^{-2}$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('SSRD_PL.png')


# Solar netherlands
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'ssrdNL', ax=ax, cmap='magma')
ax.set_title('Mean solar radiation at the surface during a Dunkelflaute [W m $^{-2}$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('SSRD_NL.png')

# Solar france
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'ssrdFR', ax=ax, cmap='magma')
ax.set_title('Mean solar radiation at the surface during a Dunkelflaute [W m $^{-2}$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('SSRD_FR.png')

# Wind 10m poland
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'ws10PL', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 10 meters during a Dunkelflaute [$m/s$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND10_PL.png')


# Wind 10m netherlands
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'ws10NL', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 10 meters during a Dunkelflaute [$m/s$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND10_NL.png')

# Wind 10m france
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'ws10FR', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 10 meters during a Dunkelflaute [$m/s$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND10_FR.png')

# Wind 100m poland
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'var_100_metre_wind_speedPL', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 100 meters during a Dunkelflaute [$m/s$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND100_PL.png')


# Wind 100m netherlands
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'var_100_metre_wind_speedNL', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 100 meters during a Dunkelflaute [$m/s$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND100_NL.png')

# Wind 100m france
fig, axes = plt.subplots(figsize=(10, 10), dpi=120)
ax = axes[0]
scatter_points_europe(DF_Data, 'var_100_metre_wind_speedFR', ax=ax, cmap='magma')
ax.set_title('Mean wind speed at 100 meters during a Dunkelflaute [$m/s$]',fontsize=12)
#ax.xticks(fontsize=23)
#ax.yticks(fontsize=23)
fig.tight_layout()
fig.savefig('WIND100_FR.png')
