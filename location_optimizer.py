#!/usr/bin/env python
# coding: utf-8


# In[10]:


import pandas as pd
from keplergl import KeplerGl
import numpy as np
import h3
from shapely.geometry import Polygon, Point
import geopandas as gpd
import streamlit as st
from streamlit_keplergl import keplergl_static
from shapely import wkt
import time

config = {'version': 'v1',
          'config': {'visState': {'filters': [],
                                  'layers': [{'id': '35wg7r',
                                              'type': 'hexagonId',
                                              'config': {'dataId': 'hex_data',
                                                         'label': 'h3_id',
                                                         'color': [248, 149, 112],
                                                         'columns': {'hex_id': 'h3_id'},
                                                         'isVisible': True,
                                                         'visConfig': {'opacity': 0.72,
                                                                       'colorRange': {'name': 'Global Warming 5',
                                                                                      'type': 'sequential',
                                                                                      'category': 'Uber',
                                                                                      'colors': ['#FFC300', '#D55D0E',
                                                                                                 '#AC1C17', '#831A3D',
                                                                                                 '#5A1846'],
                                                                                      'reversed': True},
                                                                       'coverage': 0.95,
                                                                       'enable3d': True,
                                                                       'sizeRange': [0, 500],
                                                                       'coverageRange': [0, 1],
                                                                       'elevationScale': 5,
                                                                       'enableElevationZoomFactor': True},
                                                         'hidden': False,
                                                         'textLabel': [{'field': None,
                                                                        'color': [255, 255, 255],
                                                                        'size': 18,
                                                                        'offset': [0, 0],
                                                                        'anchor': 'start',
                                                                        'alignment': 'center'}]},
                                              'visualChannels': {'colorField': {'name': 'h3_score', 'type': 'integer'},
                                                                 'colorScale': 'quantize',
                                                                 'sizeField': {'name': 'h3_score', 'type': 'integer'},
                                                                 'sizeScale': 'linear',
                                                                 'coverageField': None,
                                                                 'coverageScale': 'linear'}},
                                             {'id': 'a3so98a',
                                              'type': 'point',
                                              'config': {'dataId': 'point_data',
                                                         'label': 'Point',
                                                         'color': [16, 129, 136],
                                                         'columns': {'lat': 'lat', 'lng': 'lon', 'altitude': None},
                                                         'isVisible': True,
                                                         'visConfig': {'radius': 14.8,
                                                                       'fixedRadius': False,
                                                                       'opacity': 1,
                                                                       'outline': True,
                                                                       'thickness': 1,
                                                                       'strokeColor': [207, 220, 79],
                                                                       'colorRange': {'name': 'Global Warming',
                                                                                      'type': 'sequential',
                                                                                      'category': 'Uber',
                                                                                      'colors': ['#5A1846',
                                                                                                 '#900C3F',
                                                                                                 '#C70039',
                                                                                                 '#E3611C',
                                                                                                 '#F1920E',
                                                                                                 '#FFC300']},
                                                                       'strokeColorRange': {'name': 'Global Warming',
                                                                                            'type': 'sequential',
                                                                                            'category': 'Uber',
                                                                                            'colors': ['#5A1846',
                                                                                                       '#900C3F',
                                                                                                       '#C70039',
                                                                                                       '#E3611C',
                                                                                                       '#F1920E',
                                                                                                       '#FFC300']},
                                                                       'radiusRange': [0, 27.6],
                                                                       'filled': True},
                                                         'hidden': False,
                                                         'textLabel': [{'field': None,
                                                                        'color': [255, 255, 255],
                                                                        'size': 18,
                                                                        'offset': [0, 0],
                                                                        'anchor': 'start',
                                                                        'alignment': 'center'}]},
                                              'visualChannels': {'colorField': None,
                                                                 'colorScale': 'quantile',
                                                                 'strokeColorField': None,
                                                                 'strokeColorScale': 'quantile',
                                                                 'sizeField': {'name': 'point_score',
                                                                               'type': 'integer'},
                                                                 'sizeScale': 'sqrt'}}],
                                  'interactionConfig': {'tooltip': {'fieldsToShow': {'hex_data': [{'name': 'h3_id',
                                                                                                   'format': None},
                                                                                                  {
                                                                                                      'name': 'h3_centroid',
                                                                                                      'format': None},
                                                                                                  {'name': 'h3_score',
                                                                                                   'format': None}],
                                                                                     'point_data': [{'name': 'point_id',
                                                                                                     'format': None},
                                                                                                    {'name': 'lat',
                                                                                                     'format': None},
                                                                                                    {'name': 'lon',
                                                                                                     'format': None},
                                                                                                    {
                                                                                                        'name': 'point_score',
                                                                                                        'format': None}]},
                                                                    'compareMode': False,
                                                                    'compareType': 'absolute',
                                                                    'enabled': True},
                                                        'brush': {'size': 0.5, 'enabled': False},
                                                        'geocoder': {'enabled': False},
                                                        'coordinate': {'enabled': False}},
                                  'layerBlending': 'additive',
                                  'splitMaps': [],
                                  'animationConfig': {'currentTime': None, 'speed': 1}},
                     'mapState': {'bearing': -35.677331102854495,
                                  'dragRotate': True,
                                  'latitude': 55.91667398915577,
                                  'longitude': -3.233495830389323,
                                  'pitch': 51.034161806643965,
                                  'zoom': 10.147602307151766,
                                  'isSplit': False},
                     'mapStyle': {'styleType': 'light',
                                  'topLayerGroups': {},
                                  'visibleLayerGroups': {'label': True,
                                                         'road': True,
                                                         'border': False,
                                                         'building': True,
                                                         'water': True,
                                                         'land': True,
                                                         '3d building': False},
                                  'threeDBuildingColor': [9.665468314072013,
                                                          17.18305478057247,
                                                          31.1442867897876],
                                  'mapStyles': {}}}}


# h3_id = '8843acd819fffff'
# output_h3_id_attributes('8843acd819fffff')

# h3.h3_to_geo_boundary(h3_id, geo_json=True)

# polygon_string = 'POLYGON ((55.23774694448915 25.31287951583933, 55.23951392798415 25.30858249027833, 55.24462934439571 25.30745909858661, 55.24797838839316 25.31063268305425, 55.24621180412194 25.31493004366226, 55.24109577657453 25.31605348476041, 55.23774694448915 25.31287951583933))'

# shapely_polygon_fig = wkt.loads(polygon_string)
# shapely_polygon_fig.wkt

def output_h3_id_attributes(h3_id):
    return {
        "co_ordinates": h3.h3_to_geo(h3_id),
        "geo_boundary": Polygon(h3.h3_to_geo_boundary(h3_id, geo_json=True)).wkt,
        "parent": h3.h3_to_parent(h3_id),
        "children": h3.h3_to_children(h3_id)
    }


def generate_city_gdf(country, city, layer, config):
    city_gdf = pd.DataFrame()
    gpkg_path = "geofences/" + country + ".gpkg"
    city_gdf = gpd.read_file(gpkg_path, layer=layer)  # Layer selected after sampling on GeoPackage viewer

    temp = city_gdf[city_gdf['NAME_1'] == city]

    if len(temp.index) > 0:
        city_gdf = temp
    else:
        city_gdf = city_gdf[city_gdf['NAME_2'] == city]

    if city == 'Dubai':
        # Filter sectors (districts) closer to the coastline
        city_gdf = city_gdf[
            city_gdf['NAME_2'].isin(['Sector 1', 'Sector 2', 'Sector 3', 'Sector 4', 'Sector 5', 'Sector 6'])]
        city_gdf = city_gdf.loc[[49]]
        config['config']['mapState']['latitude'] = 25.12577627331186
        config['config']['mapState']['longitude'] = 55.20620510125014

    elif city == 'Edinburgh':
        config['config']['mapState']['latitude'] = 55.93105
        config['config']['mapState']['longitude'] = -3.31421

    elif city == 'Sydney':
        config['config']['mapState']['latitude'] = -33.88482
        config['config']['mapState']['longitude'] = 151.21492

    elif city == 'Istanbul':
        config['config']['mapState']['latitude'] = 41.11733
        config['config']['mapState']['longitude'] = 28.9703

    # Essential columns
    city_gdf = city_gdf[['NAME_0', 'NAME_1', 'NAME_2', 'geometry']]
    # Rename columns
    city_gdf.columns = ['Country', 'City', 'Sector', 'Geometry']
    city_gdf = pd.DataFrame(city_gdf)

    # multi_polygon_sector_3 = city_gdf.loc[49, 'Geometry']
    # multi_polygon_sector_3
    # mps3 = gpd.GeoSeries(multi_polygon_sector_3.geoms)
    # mps3.plot()
    return city_gdf, config


def generate_h3_df(city_gdf, resolution=7):
    h3_dict = {}
    counter = 0
    h3_df = pd.DataFrame()
    # Iterate over every row of the geo dataframe
    for _, row in city_gdf.iterrows():
        # Parse out info from columns of row
        country = row.Country
        city = row.City
        district_multipolygon = row.Geometry
        district_sector = row.Sector
        # Convert multi-polygon into list of polygons
        district_polygon = list(district_multipolygon.geoms)
        for polygon in district_polygon:
            # Convert Polygon to GeoJSON dictionary
            poly_geojson = gpd.GeoSeries([polygon]).__geo_interface__
            # Parse out geometry key from GeoJSON dictionary
            poly_geojson = poly_geojson['features'][0]['geometry']
            # Fill the dictionary with Resolution 10 H3 Hexagons
            h3_hexes = h3.polyfill_geojson(poly_geojson, resolution)
            for h3_hex in h3_hexes:
                h3_geo_boundary = h3.h3_to_geo_boundary(h3_hex, geo_json=False)
                h3_centroid = h3.h3_to_geo(h3_hex)
                # Append results to dataframe
                h3_dict[counter] = [
                    country,
                    city,
                    district_sector,
                    h3_hex,
                    h3_geo_boundary,
                    h3_centroid
                ]
                counter += 1

    # Write dictionary to DataFrame
    h3_df[['country', 'city', 'sector', 'h3_id', 'h3_geo_boundary', 'h3_centroid']] = pd.DataFrame.from_dict(
        data=h3_dict, orient='index')
    return h3_df[~h3_df.h3_id.isin(['8743a132cffffff', '8743a1acaffffff', '8743a1ac9ffffff', '8743a1aceffffff',
                                    '8743a1ac8ffffff', '8743acd94ffffff'])]


def generate_point_df(h3_df, empty_hex_rate=0.5):
    hex_size = len(h3_df.index)
    num_points_for_hex = np.random.randint(1, 6, size=hex_size)
    num_points_for_hex[:int(hex_size * empty_hex_rate)] = 0
    np.random.shuffle(num_points_for_hex)

    counter = 0
    point_dict = {}
    point_df = pd.DataFrame()
    list_h3_score = []
    for i, idx in enumerate(h3_df.index):
        h3_vertices = h3_df.loc[idx, 'h3_geo_boundary']
        h3_id = h3_df.loc[idx, 'h3_id']
        h3_score = 0
        for j in range(num_points_for_hex[i]):
            point_id = 'P' + str(counter)
            point_score = np.random.randint(0, 101)
            h3_score += point_score
            conv_hull_coeffs = np.random.rand(6, 1)
            conv_hull_coeffs = conv_hull_coeffs / sum(conv_hull_coeffs)
            # print(conv_hull_coeffs)
            point_centroid = sum(h3_vertices * conv_hull_coeffs)
            lat = point_centroid[0]
            lon = point_centroid[1]
            point_dict[counter] = [point_id, point_score, lat, lon, h3_id]
            counter += 1
        list_h3_score.append(h3_score)

    list_h3_score = np.array(list_h3_score)
    h3_df['h3_score'] = list_h3_score
    if len(point_dict) > 0:
        point_df[['point_id', 'point_score', 'lat', 'lon', 'h3_id']] = pd.DataFrame.from_dict(data=point_dict,
                                                                                              orient='index')
    return h3_df, point_df


# ![image.png](attachment:image.png)

# In[29]:
sidebar = st.sidebar

#"""
#   A GUI implementation for toy example of EV Charge Station (EVCS).\n
#   It works with mock data.
#"""

sidebar.write(
    """
       A GUI implementation for solving Location Selection Problem.\n
       It works with mock data.
    """
)

country = sidebar.selectbox("Select a Country", ["United Kingdom", "United Arab Emirates", "Australia", "Turkey"])
city = 'Edinburgh'
layer = 'gadm36_GBR_2'

if country == "United Kingdom":
    country = 'UK'
    city = sidebar.selectbox("Select a City", ["Edinburgh"])
    layer = 'gadm36_GBR_2'
elif country == "United Arab Emirates":
    country = 'UAE'
    city = sidebar.selectbox("Select a City", ["Dubai"])
    layer = 'gadm36_ARE_2'
elif country == "Australia":
    country = 'AUS'
    city = sidebar.selectbox("Select a City", ["Sydney"])
    layer = 'gadm36_AUS_2'
elif country == "Turkey":
    country = 'TUR'
    city = sidebar.selectbox("Select a City", ["Istanbul"])
    layer = 'gadm36_TUR_2'


sensitivty = sidebar.slider("Select a Resolution Level for Hexagons", min_value=7, max_value=10)
pick_top_n_score = sidebar.slider("Select Number of Locations", min_value=1, max_value=20) #for EVCS Installation
# Set parameters
np.random.seed(seed=0)
#country = 'UK'  # UAE, UK, AUS, TUR
#city = 'Edinburgh'
#layer = 'gadm36_GBR_2'  # ARE for UAE, GBR for UK, AUS for AUS, TUR for TUR
#sensitivty = 7
empty_hex_rate = 0.5
cover_rate = 0.6
#pick_top_n_score = 10
# pick n top score parametresi ekleyelim.

city_gdf, config = generate_city_gdf(country, city, layer, config)
h3_df = generate_h3_df(city_gdf, sensitivty)
h3_df, point_df = generate_point_df(h3_df, empty_hex_rate)
w1 = KeplerGl(height=400, data={'hex_data': h3_df[['h3_id', 'h3_centroid', 'h3_score']]}, config=config)
w1.add_data(data=point_df[['point_id', 'point_score', 'lat', 'lon']], name='point_data')
# w1


# In[27]:


h3_pick_n_df = h3_df.sort_values(by='h3_score', ascending=False).iloc[:pick_top_n_score, :]
w1 = KeplerGl(height=400, data={'hex_data': h3_pick_n_df[['h3_id', 'h3_centroid', 'h3_score']]}, config=config)
w1.add_data(data=point_df[['point_id', 'point_score', 'lat', 'lon']], name='point_data')
keplergl_static(w1)
# w1
st.title("Facility Location Optimizer")
st.write("Helps you make your decisions optimally")

