# Tremor inversion 

This repository contains the code used in [van der Laat *et al.* (in prep.)]() to invert tremor using the forward model from Girona *et al.* (2019) 

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

# Run the example

## Get the data

    python ./example/get_data.py

## Extract the observations

    ti-extract ./example/config.yml

## Genetic Algorith

Test:

    ti-ga-multi-test ./example/config.yml UWE

Run all:

    ti-ga-multi ./example/config.yml UWE
