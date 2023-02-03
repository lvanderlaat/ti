# TODO
* Add script for inventory and channel `csv` file.
* Add script `mass_download`

# Tremor inversion 
This repository contains the code used in [van der Laat *et al.* (in prep.)]() to invert tremor using the forward model from Girona *et al.* (2019) 

# Installation

Download a zip file of the code or clone it (HTTP):

    git clone https://github.com/lvanderlaat/tremor_inversion.git

or (SSH):

    git clone git@github.com:lvanderlaat/tremor_inversion.git

Go in the repository directory

    cd tremor_inversion

and create a `conda` environment named `ti` and install the package and its dependencies:
    
    conda env create -f environment.yml

Activate the environment

    conda activate ti

and install this package:

    pip install -e .

# Run the example

This repository contains a minimal working example that you can run to learn how to use this program:

    cd example
    jupyter-lab example.ipynb
