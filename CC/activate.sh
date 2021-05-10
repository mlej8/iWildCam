#!/bin/bash


# This script creates the iWildCam kernel
source iWildCam-env/bin/activate

# When you have created a virtual environment, you would realize that the virtual environment is separate from your Jupyter Notebook. We need to set up a few things before we could have our virtual environment in the Jupyter Notebook. First, activate your virtual environment and run this code.
pip install --user ipykernel

# We need to manually add the kernel if we want to have the virtual environment in the Jupyter Notebook. That is why we need to add it by running this code.
python -m ipykernel install --user --name=iWildcam-env