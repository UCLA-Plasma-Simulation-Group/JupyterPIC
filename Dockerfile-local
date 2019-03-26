FROM jupyter/scipy-notebook

MAINTAINER Benjamin J. Winjum <bwinjum@ucla.edu>
#With grateful acknowledgements to the Jupyter Project <jupyter@googlegroups.com> for Jupyter
#And to the Particle-in-Cell and Kinetic Simulation Software Center for OSIRIS:

USER root

#
# NECESSITIES FOR CODE COMPILING / RUNNING
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
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#
# C kernel
#
RUN pip install --no-cache-dir jupyter-c-kernel && \
    install_c_kernel --user

#
# Java kernel
#
# Install Java.
RUN \
  apt-get update && \
  apt-get install -y openjdk-11-jre && \
  rm -rf /var/lib/apt/lists/*

# Define commonly used JAVA_HOME variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64

# Get IJava
RUN curl -L https://github.com/SpencerPark/IJava/releases/download/v1.2.0/ijava-1.2.0.zip > ijava-kernel.zip

# Unpack and install the kernel
RUN unzip ijava-kernel.zip -d ijava-kernel \
  && cd ijava-kernel \
  && python3 install.py --sys-prefix

#
# Complete setting up the environment for OSIRIS and JupyterPIC
# 

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

USER $NB_USER

