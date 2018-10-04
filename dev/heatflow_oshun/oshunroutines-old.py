#!/usr/bin/python

import sys
import h5py
import numpy as np

def pullData(directory,quantity,time):
    time = int(time)
    if (time < 1e1 and time >= 0):
        timestr = '0000' + str(time);
    elif (time < 1e2 and time >= 1e1):
        timestr = '000' + str(time);
    elif (time < 1e3 and time >= 1e2):
        timestr = '00' + str(time);
    elif (time < 1e4 and time >= 1e3):
        timestr = '0' + str(time);
    else:
        timestr = str(time);

    if (quantity == 'Ex' or quantity == 'Ey' or quantity == 'Ez' or
            quantity == 'Bx' or quantity == 'By' or quantity == 'Bz'):

        subdir = 'OUTPUT/FLD/' + quantity;
        typeofquantity = 'FLD';
        filename = directory + '/'  + subdir  + '/'  + quantity  +  '_'  + timestr  + '.h5';

        outfile = h5py.File(filename, 'r');

    elif (quantity == 'n' or quantity == 'Jx' or quantity == 'T' or
                quantity == 'Qx' or quantity == 'VNx' or
                quantity == 'ni' or quantity == 'Ux'  or
                quantity == 'Ti' or quantity == 'Z'):

        subdir = 'OUTPUT/MOM/' + quantity;
        typeofquantity = 'FLD';
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';

        outfile = h5py.File(filename, 'r');

    elif (quantity == 'f0-x' or quantity == 'f10-x' or quantity == 'f11-x' or quantity == 'f20-x' or quantity == 'fl0-x' ):

        subdir = 'OUTPUT/DISTR/' + quantity;
        typeofquantity = 'f';
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');

    elif (quantity == 'px-x' ):

        subdir = 'OUTPUT/DISTR/' + quantity;
        typeofquantity = 'fulldist';
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');
        
    elif (quantity == 'pxpy-x' ):

        subdir = 'OUTPUT/DISTR/' + quantity;
        typeofquantity = 'fulldist';
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');
            
    elif (quantity == 'df' ):
    
        quantity = 'px-x';
        simtime1d, axis1, axis2, px = oshun.pullData(fulldir,quantity,time1d);
        
        subdir = 'OUTPUT/DISTR/' + quantity;
        typeofquantity = 'f';
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');

    infokeys=outfile['/'].attrs.keys()
    infovalues=outfile['/'].attrs.values()

    axes = outfile.get('AXIS')
    data = outfile.get(quantity)
    time = infovalues[1][0]
    
    out = np.array(data)
    numaxes = len(outfile['/AXIS'].keys())

    if typeofquantity == 'FLD':

        axis1lims = axes.get('AXIS1')
        axis1out = np.array(axis1lims)
        dx = (axis1lims[1] - axis1lims[0])/data.shape[0]

        axis1 = np.linspace(axis1lims[0]+dx/2, axis1lims[1]-dx/2, data.shape[0])

        axis2 = 0

    elif typeofquantity == 'f':
        
        axis1lims = axes.get('AXIS1')
        axis1out = np.array(axis1lims)
        dx = (axis1lims[1] - axis1lims[0])/data.shape[0]

        axis1 = np.linspace(axis1lims[0]+dx/2, axis1lims[1]-dx/2, data.shape[0])

        axis2lims = axes.get('AXIS2')
        axis2out = np.array(axis2lims)
        dx = (axis2lims[1] - axis2lims[0])/data.shape[1]

        axis2 = np.linspace(axis2lims[0]+dx/2, axis2lims[1]-dx/2, data.shape[1])

    elif typeofquantity == 'fulldist':

        axis1lims = axes.get('AXIS1')
        axis1out = np.array(axis1lims)
        dx = (axis1lims[1] - axis1lims[0])/data.shape[0]

        axis1 = np.linspace(axis1lims[0]+dx/2, axis1lims[1]-dx/2, data.shape[0])

        axis2lims = axes.get('AXIS2')
        axis2out = np.array(axis2lims)
        dx = (axis2lims[1] - axis2lims[0])/data.shape[1]

        axis2 = np.linspace(-(axis2lims[1]-dx/2), axis2lims[1]-dx/2, data.shape[1])

        
    outfile.close()
    
    return time, axis1, axis2, out

def getxt(directory,quantity,timerange):
    simtime, axis1, axis2, data = pullData(directory,quantity,0)
    
    dataoveralltime = np.zeros((timerange.shape[0],axis1.shape[0]))
    timeaxis = []
    datamax = []
    it = 0
    for time in timerange:
        simtime, axis1, axis2, data = pullData(directory,quantity,time)
        dataoveralltime[it,:] = data
        timeaxis.append(simtime)
        datamax.append(max(abs(data)))
        it = it + 1
    
    return timeaxis, axis1, datamax, dataoveralltime
    
    