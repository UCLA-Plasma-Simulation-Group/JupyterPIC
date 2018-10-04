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

        subdir = 'output/fields/' + quantity;
        
        filename = directory + '/'  + subdir  + '/'  + quantity  +  '_'  + timestr  + '.h5';

        outfile = h5py.File(filename, 'r');

    elif (quantity == 'n' or quantity == 'T' or
          quantity == 'Jx' or quantity == 'Jy' or quantity == 'Jz' or
          quantity == 'Qx' or quantity == 'VNx' or 
          quantity == 'Qy' or quantity == 'VNy' or
          quantity == 'ni' or quantity == 'Ux'  or 
          quantity == 'Ti' or quantity == 'Z'):

        subdir = 'output/moments/' + quantity;
        
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';

        outfile = h5py.File(filename, 'r');

    elif (quantity == 'dt'):

        subdir = 'output/fields/' + 'Ex';
        
        filename = directory + '/'  + subdir  + '/'  + 'Ex'  +  '_'  + timestr  + '.h5';

        outfile = h5py.File(filename, 'r');        

    elif (quantity == 'f0' or quantity == 'f10' or quantity == 'f11' or quantity == 'f20' or quantity == 'fl0' ):

        subdir = 'output/distributions/' + quantity;
        
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');

    elif (quantity == 'px' ) or (quantity == 'py' ) or (quantity == 'pz' ):

        subdir = 'output/distributions/' + quantity;
        
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');
                
    elif (quantity == 'pxpy' or (quantity == 'pxpz' ) or (quantity == 'pypz' ) ):

        subdir = 'output/distributions/' + quantity;
        
        filename = directory + '/' + subdir + '/' + quantity + '_s0_' + timestr + '.h5';
        outfile = h5py.File(filename, 'r');
        
    elif (quantity == 'prtx' or quantity == 'prtpx' or quantity == 'prtpy' or
         quantity == 'prtpz'):

        subdir = 'output/particles/' + quantity;

        filename = directory + '/'  + subdir  + '/'  + quantity  +  '_'  + timestr  + '.h5';

        outfile = h5py.File(filename, 'r');

    if (quantity == 'dt'): 
        data = np.array(outfile.get('Ex'))
        data = np.ones(data.shape)*outfile['Ex'].attrs.get('dt');
        time = outfile['Ex'].attrs.get('Time (c/\omega_p)')
        timeps = outfile['Ex'].attrs.get('Time (ps)')
        numaxes = 1
    else:
        data = np.array(outfile.get(quantity))
        time = outfile[quantity].attrs.get('Time (c/\omega_p)')
        timeps = outfile[quantity].attrs.get('Time (ps)')
        numaxes = len(data.shape)
    
    axes = []

    for a in range(1,numaxes+1,1):
        axisstr = 'Axes/Axis' + str(a)
        tmp = np.array(outfile.get(axisstr))
        axes.append(tmp)
    
    outfile.close()
    
    # print data

    return time, timeps, axes, data

def getxt(directory,quantity,timerange):
    simtime, realtime, axes, data = pullData(directory,quantity,0)
    
    if (quantity == 'prtx'):
#         print axes[0].shape
        axes[0] = axes[0][axes[0].shape[0]/2:]
#         print axes
    
    
    
    dataoveralltime = np.zeros((timerange.shape[0],axes[0].shape[0]))
    timeaxis = []
    datamax = []
    it = 0
    for time in timerange:
        simtime, realtime, axes, data = pullData(directory,quantity,time)
        dataoveralltime[it,:] = data
        timeaxis.append(simtime)
        datamax.append(max(abs(data)))
        it = it + 1
    
    return timeaxis, axes, datamax, dataoveralltime
    
    