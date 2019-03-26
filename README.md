[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/UCLA-Plasma-Simulation-Group/JupyterPIC/master)

# Jupyter notebooks for educational plasma physics simulations with PIC

This repository contains educational Jupyter notebooks for plasma physics.  The notebooks use particle-in-cell simulations with the 1D PIC code OSIRIS to illustrate:
* Dispersion relation for electron plasma waves
* Dispersion relation for light waves
* Light waves traveling from vacuum into plasma (with either gradual or sharp gradient)
* X- and O-mode dispersion
* Propagating X-waves
* R- and L-waves
* Faraday rotation
* Velocities:  introduction to phase and group velocity (no simulations)
* Plasma Wake Field Accelerator (simulated by 3D quasi-static PIC code [QuickPIC](https://github.com/UCLA-Plasma-Simulation-Group/QuickPIC-OpenSource.git))

We welcome contributions and ideas:  please email us at picksc.org@gmail.com

## To run this on your local computer:

* Docker must be installed ([Docker site](https://www.docker.com/))
* Clone this repository
* Navigate into the JupyterPIC directory
* Execute `./runosdocker`
* This will output several lines of text to your terminal, including a web URL
* Paste the web URL into a web browser to start the Jupyter environment
* Within Jupyter, open the *.ipynb files inside any of the "notebooks" directories 
* To quit, "Logout" of the Jupyter notebook, close your browser windows, and type Ctrl-C twice in your terminal window.

The above commands will write simulation output into the project directories.  If you would like to keep a fresh copy of this original GitHub repository, it is important that you copy this JupyterPIC directory into a separate location on your computer for simulation purposes and personal file modifications.

## Running notebooks under Windows:

The docker environment also works under windows if you can install Docker on your Windows system.  Installing docker requires very recent versions of Windows 10 (2018 or later), but if you have a newer (i.e., purchased after 2017) computer you should be able install and run the Docker container needed to run these notebooks.  Then, you must enable disk sharing on the directory where this GitHub repo is located.  (i.e., C:, D: or G:)  To start the Docker container, first download it from the Dockerhub

docker pull picksc/jupyterpic

Then open PowerShell in Windows, and launch the docker container.

docker run -v ${PWD}:/home/jovyan -p 8888:8888 picksc/jupyterpic

