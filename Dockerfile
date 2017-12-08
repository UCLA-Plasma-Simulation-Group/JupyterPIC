FROM jupyter/datascience-notebook

MAINTAINER Benjamin J. Winjum <bwinjum@ucla.edu>
#With grateful acknowledgements to the Jupyter Project <jupyter@googlegroups.com> for Jupyter
#And to the Particle-in-Cell and Kinetic Simulation Software Center for OSIRIS:

USER root

#
# OSIRIS
#
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    libopenmpi-dev \
    libhdf5-openmpi-dev \
    gfortran \
    openmpi-bin \
    openmpi-common \
    openmpi-doc \
    gcc-4.8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV H5_ROOT /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib

RUN mkdir /usr/local/osiris
ENV PATH $PATH:/usr/local/osiris
ENV PYTHONPATH $PYTHONPATH:/usr/local/osiris
COPY osiris-1D.e /usr/local/osiris/osiris-1D.e
COPY osiris.py /usr/local/osiris/osiris.py
COPY combine_h5_util_1d.py /usr/local/osiris/combine_h5_util_1d.py
COPY analysis.py /usr/local/osiris/analysis.py
COPY h5_utilities.py /usr/local/osiris/h5_utilities.py
COPY str2keywords.py /usr/local/osiris/str2keywords.py
WORKDIR work
COPY osiris-class osiris-class
RUN chmod -R 711 /usr/local/osiris/osiris-1D.e

USER $NB_USER
