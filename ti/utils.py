#!/usr/bin/env python


"""
Some utils (I/O, etc.)
"""


# Python Standard Library
import argparse
import datetime
import json
import os

# Other dependencies
import pandas as pd
import numpy as np

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def scan_dir(wfs_dir, datetime_fmt='%Y%m%dT%H%M%SZ'):
    filenames = os.listdir(wfs_dir)
    data = []
    for filename in filenames:
        network, station, location, cha_date, extension = filename.split('.')
        channel, starttime, endtime = cha_date.split('__')
        datum = dict(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=datetime.datetime.strptime(starttime, datetime_fmt),
            endtime=datetime.datetime.strptime(endtime, datetime_fmt),
            filename=filename
        )
        data.append(datum)
    df = pd.DataFrame(data)
    df['_starttime'] = df.starttime
    df.index = df._starttime
    df.sort_index(inplace=True)

    df['stacha'] = list(zip(df.station, df.channel))
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('configfile', help='Configuration file path')
    return parser.parse_args()


def datetime_id(dt):
    return dt.strftime('%Y%m%d_%H%M%S')


def create_folder(output_dir, name, overwrite):
    folderpath = os.path.join(output_dir, name)

    if not overwrite:
        folderpath = folderpath + '_' + datetime_id(datetime.datetime.now())

    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    return folderpath


def write_conf(c, folderpath):
    with open(os.path.join(folderpath, 'config.json'), 'w') as f:
        json.dump(c, f, indent=4, default=str)


if __name__ == '__main__':
    dt = datetime.datetime.now()
    dt_id = datetime_id(dt)
    print(dt_id)
