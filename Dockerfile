FROM jupyter/scipy-notebook

MAINTAINER Benjamin J. Winjum <bwinjum@ucla.edu>
#With grateful acknowledgements to the Jupyter Project <jupyter@googlegroups.com> for Jupyter
#And to the Particle-in-Cell and Kinetic Simulation Software Center for OSIRIS:

USER root

#
# OSIRIS
#
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    openssh-client \
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
RUN chmod 777 Landau-Damping
RUN chmod 777 buneman
RUN chmod 777 driven_waves
RUN chmod 777 heatflow_oshun
RUN chmod 777 iaw-fluid-theory
RUN chmod 777 interactive-theory
RUN chmod 777 quickpic_pwfa
RUN chmod 777 weibel

WORKDIR ..

USER $NB_USER 
