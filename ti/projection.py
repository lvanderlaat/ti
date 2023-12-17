#!/usr/bin/env python


"""
"""


# Python Standard Library

# Other dependencies
from pyproj import Proj

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'lvmzxc@gmail.com'


def transform(x, y, origin, inverse=False):
    proj = Proj(
        proj='aeqd',
        lon_0=origin[0],
        lat_0=origin[1],
        datum='WGS84',
        units='m'
    )
    return proj(x, y, inverse=inverse)


if __name__ == '__main__':
    pass
