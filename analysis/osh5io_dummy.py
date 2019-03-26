#!/usr/bin/env python

"""dummy module for testing purpose."""


import h5py
import os
from osh5def import *
from osunit import *


def read_h5(filename, path=None):
    fn = filename if not path else path + '/' + filename
    print('reading from ' + fn)
    return 1


def write_h5(data, filename=None, path=None, dataset_name=None, write_data=True):
    if dataset_name is not None:
        current_name_attr = dataset_name
    else:
        current_name_attr = "Data"

    fname = path if path else ''
    if filename is not None:
        fname += filename
    else:
        fname += current_name_attr + '.h5'
    print('writing to '+fname)
