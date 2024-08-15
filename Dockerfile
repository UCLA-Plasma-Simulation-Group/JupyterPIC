# FROM jupyter/scipy-notebook
FROM quay.io/jupyter/scipy-notebook@sha256:3b8d0f5253e3acb5395bee6638bf51762b84e159f59b27c46d0b9cd27fac2306

MAINTAINER Benjamin J. Winjum <bwinjum@ucla.edu>
#With grateful acknowledgements to the Jupyter Project <jupyter@googlegroups.com> for Jupyter
#And to the Particle-in-Cell and Kinetic Simulation Software Center for OSIRIS:

USER root

# This is to solve an issue with permissions when running under Windows
#RUN conda install -c anaconda jupyter_client=5.3.1

#
# OSIRIS
#

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    gfortran \
    openmpi-bin \
    openmpi-common \
    openmpi-doc \
    gcc \
    openssh-client \
    libopenmpi-dev \
    libhdf5-openmpi-dev \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get install -y python
# RUN apt-get install -y python-is-python3

# ************************************************************************
# ************************************************************************
# Install json-Fortran for QuickPIC
RUN pip install FoBiS.py
RUN conda install --channel conda-forge json-fortran
# ************************************************************************
# ************************************************************************

# *************************************************************************
# Configure + Build HDF5 (with MPI support)
#
# WORKDIR /root
# RUN wget https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.5.tar
# RUN tar xvf hdf5-1.10.5.tar
# WORKDIR /root/hdf5-1.10.5
# RUN ./configure --enable-fortran --enable-parallel --enable-shared --prefix=/osiris_libs/hdf5 CC=mpicc FC=mpif90 F90=mpif90
# RUN make
# RUN make install
# **************************************************************************


# ENV H5_ROOT /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib
ENV H5_ROOT /usr/lib/aarch64-linux-gnu/hdf5/openmpi/
ENV OMPI_MCA_btl_vader_single_copy_mechanism none

RUN mkdir /usr/local/osiris
RUN mkdir /usr/local/beps
RUN mkdir /usr/local/quickpic
RUN mkdir /usr/local/oshun
ENV PATH $PATH:/usr/local/osiris:/usr/local/beps:/usr/local/quickpic:/usr/local/oshun
ENV PYTHONPATH $PYTHONPATH:/usr/local/osiris:/usr/local/quickpic:/usr/local/oshun
COPY bin/osiris-1D.e /usr/local/osiris/osiris-1D.e
COPY bin/osiris-2D.e /usr/local/osiris/osiris-2D.e
COPY bin/upic-es.out /usr/local/beps/upic-es.out
COPY bin/qpic.e /usr/local/quickpic/qpic.e
COPY bin/oshun.e /usr/local/oshun/oshun.e
COPY analysis/osiris.py /usr/local/osiris/osiris.py
COPY analysis/combine_h5_util_1d.py /usr/local/osiris/combine_h5_util_1d.py
COPY analysis/combine_h5_util_2d.py /usr/local/osiris/combine_h5_util_2d.py
COPY analysis/combine_h5_util_2d_true.py /usr/local/osiris/combine_h5_util_2d_true.py
COPY analysis/combine_h5_util_3d.py /usr/local/osiris/combine_h5_util_3d.py
COPY analysis/analysis.py /usr/local/osiris/analysis.py
COPY analysis/h5_utilities.py /usr/local/osiris/h5_utilities.py
COPY analysis/str2keywords.py /usr/local/osiris/str2keywords.py
COPY analysis/quickpic.py /usr/local/quickpic/quickpic.py
COPY analysis/oshunroutines.py /usr/local/oshun/oshunroutines.py
COPY analysis/heatflowroutines.py /usr/local/oshun/heatflowroutines.py
COPY analysis/osh5def.py /usr/local/oshun/osh5def.py
COPY analysis/osh5gui.py /usr/local/oshun/osh5gui.py
COPY analysis/osh5io.py /usr/local/oshun/osh5io.py
COPY analysis/osh5io_dummy.py /usr/local/oshun/osh5io_dummy.py
COPY analysis/osh5utils.py /usr/local/oshun/osh5utils.py
COPY analysis/osh5vis.py /usr/local/oshun/osh5vis.py
COPY analysis/osh5visipy.py /usr/local/oshun/osh5visipy.py
RUN chmod -R 711 /usr/local/osiris/osiris-1D.e
RUN chmod -R 711 /usr/local/osiris/osiris-2D.e
RUN chmod -R 711 /usr/local/beps/upic-es.out
RUN chmod -R 711 /usr/local/quickpic/qpic.e
RUN chmod -R 711 /usr/local/oshun/oshun.e

WORKDIR work
COPY notebooks notebooks
RUN chmod 777 notebooks
WORKDIR notebooks
RUN chmod 777 electron-plasma-wave-dispersion
RUN chmod 777 faraday-rotation
RUN chmod 777 light-wave-dispersion
RUN chmod 777 light-wave-vacuum-into-plasma
RUN chmod 777 r-and-l-mode-dispersion
RUN chmod 777 two-stream
RUN chmod 777 velocities
RUN chmod 777 x-and-o-mode-dispersion
RUN chmod 777 x-mode-propagation

WORKDIR ..
COPY dev dev
RUN chmod 777 dev
WORKDIR dev
RUN chmod 777 Forslund-Kindel-Lindman-1975
RUN chmod 777 Landau-Damping
RUN chmod 777 Tajima-Dawson-1979
RUN chmod 777 buneman
RUN chmod 777 driven_waves
RUN chmod 777 heatflow_oshun
RUN chmod 777 iaw-fluid-theory
RUN chmod 777 interactive-theory
RUN chmod 777 quickpic_pwfa
RUN chmod 777 weibel
RUN chmod 777 RPA
RUN chmod 777 SBS
RUN chmod 777 TPD
RUN chmod 777 forslund-SRS
RUN chmod 777 grid-instability
RUN chmod 777 single-particle
RUN chmod 777 Leap-Frog

WORKDIR ..
COPY notebooks-260 notebooks-260
RUN chmod 777 notebooks-260
WORKDIR notebooks-260
RUN chmod 777 LWFA-Workbook-1-Tajima-Dawson
RUN chmod 777 Single-Particle-Workbook
RUN chmod 777 PWFA

WORKDIR ..
# COPY NERS-574 NERS-574
# RUN chmod 777 NERS-574
# WORKDIR NERS-574
# RUN chmod 777 faraday-rotation
# RUN chmod 777 light-wave-dispersion
# RUN chmod 777 light-wave-vacuum-into-plasma
# RUN chmod 777 r-and-l-mode-dispersion
# RUN chmod 777 velocities
# RUN chmod 777 x-and-o-mode-dispersion
# RUN chmod 777 x-mode-propagation
# RUN chmod 777 LWFA-Workbook-1-Tajima-Dawson
# RUN chmod 777 LWFA-Basic-Notebook

WORKDIR ..

USER $NB_USER 
# RUN pip install nbgitpuller==1.2.1 jupyter-resource-usage "matplotlib<3.9.0"
RUN pip install nbgitpuller==1.2.1 jupyter-resource-usage matplotlib==3.2.0
