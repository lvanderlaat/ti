# Tremor inversion 

This repository contains the code used in [van der Laat *et al.* (in review)]() to invert tremor using the forward model from Girona *et al.* (2019) 

# Installation

Clone the code, then

    cd ti
    conda create -n ti
    conda activate ti
    conda install -c anaconda pandas
    conda install -c conda-forge obspy
    conda install -c conda-forge xarray dask netCDF4 bottleneck
    conda install -c numba numba
    conda install scikit-image
    conda install -c conda-forge pyproj
    pip install -U pymoo
    pip install -e .

# Example tutorial

## Get the data

Currently, the code works with data obtained through a FDSN client (IRIS, SeisComp):


```python
from obspy.clients.fdsn.client import Client
from obspy import UTCDateTime

client = Client(base_url='IRIS')

```

The `base_url` can also be your SeisComp server address, e.g. `https://{ip}:{port}`.

We need an inventory with geographic and response data from the stations

```python
from obspy import UTCDateTime

inventory = client.get_stations(
    starttime=UTCDateTime(2018, 1, 1),
    endtime=UTCDateTime(2019, 1, 1),
    network='HV',
    station='UWE,UWB',
    channel='HHZ',
    level='channel'
)

inventory.write('example/inventory.xml', format='STATIONXML')
```

We will create a simple `csv` file with the station information. First, we create a `pandas.DataFrame` from the `obspy.Inventory`:

```python
import ti

df = ti.utils.inventory_to_dataframe(inventory)
```

and then we transform the coordinates to an Cartesian system in meters. In this case we set the origin of the Cartesian system at the position of the Halema\`uma\`u crater:

```python
df['x'], df['y'] = ti.projection.transform(
    df.longitude,
    df.latitude,
    (-155.280844, 19.405192),
    inverse=False
)
```

We save the `csv` file:

```python
df.to_csv('example/channels.csv', index=False)
```

And finally we download the waveforms:


```python
from obspy.clients.fdsn.mass_downloader import (
    CircularDomain, Restrictions, MassDownloader
)

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
```

## Extract the observations

We extract spectrograms.

We use the following parameters from the configuration file:

| Parameter | Description | Units/format |
| --------- | ----------- | ------------ |
| `io.output_dir` | Output directory | |
| `io.overwrite` | Whether to overwrite old data | |
| `max_workers` | Number of parallel processes to run | |
| `source.wfs_dir` | Directory to the raw waveform files | |
| `source.inventory_file` | Inventory file | `STATIONXML` |
| `source.channels_csv` | Channels `csv` file create | |
| `source.starttime`/`endtime` | Start and end times to extract | `%Y-%m-%d %H:%M:%S` |
| `source.sampling_rate` | Original sampling rate of the data | [Hz] |
| `preprocess.freqmin`/`freqmax` | Filter frequency limits | [Hz] |
| `model.tau` | Duration of the window of simulation | [s] |
| `model.max_freq` | Maximum frequency we will simulate. Also, will determine the decimation factor | [Hz] |
| `window.length` | Observation window length, the resulting spectrum will be the median of all subwindows of length `model.tau` that can fit in `window.length` | [s] |
| `window.overlap` | Fraction of overlap between windows | 0 - 1 |
| `window.pad` | Tukey window pad (`alpha`) | 0 - 1 |


Then run the script:

```bash
ti-extract ./example/config.yml
```

This creates a new folder `{output_dir}/SSAM` which contains:


| File | Description |
| ---- | ----------- |
| `config.json` | A copy of the configuration file (we use `json` for done processes, `yml` for new processes) |
| `data.nc` | a `netCDF` file containing the data |
| `fig/` | a subfolder with simple figures of the spectrograms, one per station and one stack |


## Genetic Algorith

Test:

    ti-ga-multi-test ./example/config.yml UWE

Run all:

    ti-ga-multi ./example/config.yml UWE
