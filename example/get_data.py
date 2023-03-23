#!/usr/bin/env python


"""
Downloads data for running example.
"""

import ti

from obspy.clients.fdsn.client import Client
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain, Restrictions, MassDownloader
)
from obspy import UTCDateTime
from pyproj import Proj, transform


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


client = Client('IRIS')

inventory = client.get_stations(
    starttime=UTCDateTime(2018, 1, 1),
    endtime=UTCDateTime(2019, 1, 1),
    network='HV',
    station='UWE,UWB',
    channel='HHZ',
    level='channel'
)

inventory.write('example/inventory.xml', format='STATIONXML')

df = ti.utils.inventory_to_dataframe(inventory)

# Geographic to Cartesian coordinates
proj1 = Proj(init='epsg:4326')
proj2 = Proj(init='epsg:26961')
df['x'], df['y'] = transform(
    proj1, proj2, df.longitude.values, df.latitude.values
)
df.to_csv('example/channels.csv', index=False)

mdl = MassDownloader(providers=['IRIS'])

mdl.download(
    CircularDomain(
        latitude=None,
        longitude=None,
        minradius=None,
        maxradius=None
    ),
    Restrictions(
        starttime=UTCDateTime(2018, 5, 11),
        endtime=UTCDateTime(2018, 5, 13),
        network=','.join(df.network.unique()),
        station=','.join(df.station.unique()),
        location='*',
        channel=','.join(df.channel.unique()),
        chunklength_in_sec=86400,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
    ),
    threads_per_client=10,
    mseed_storage='example/MSEED',
    stationxml_storage='example/stations'
)
