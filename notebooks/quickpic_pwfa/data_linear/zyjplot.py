# in the output data folder. For testing purpose
import os
import shutil
import subprocess
import IPython.display
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from h5_utilities import *
from analysis import *
from scipy.optimize import fsolve


def QEP_plot(filename,figure_title='',xlabel='',ylabel='',datamin=-1,datamax=1):
    f=h5py.File(filename,'r')
    positionOf_=filename.find('_')
    firstPartOfFilename = filename[0:positionOf_]
    dataset = f['/'+firstPartOfFilename]
    data = dataset[...]
    xaxis=f['/AXIS/AXIS1'][...]
    yaxis=f['/AXIS/AXIS2'][...]
    
    # Only plot the middle(Along x) part of data. Shift the center of x to 0 and decrease the length of x by 1/2 
    x=np.linspace((xaxis[0]-xaxis[1]/2)/2,(xaxis[1]/2)/2,int(data.shape[0]/2)) 
    y=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 

    data = data.transpose()
    datamiddle = data[63:191,:] 

    plt.axis([ y.min(), y.max(),x.min(), x.max()])
    plt.title(figure_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.pcolor(y,x,datamiddle,vmin=datamin,vmax=datamax)
    plt.colorbar()
    plt.show()
    return

def Ez_plot(filename,figure_title='',xlabel='',ylabel_left='',ylabel_right='',datamin=-1,datamax=1,lineout_position=0):
    f=h5py.File(filename,'r')
    positionOf_=filename.find('_')
    firstPartOfFilename = filename[0:positionOf_]
    dataset = f['/'+firstPartOfFilename]
    data = dataset[...]
    xaxis=f['/AXIS/AXIS1'][...]
    yaxis=f['/AXIS/AXIS2'][...]
    
    # Only plot the middle(Along x) part of data. Shift the center of x to 0 and decrease the length of x by 1/2 
    x=np.linspace((xaxis[0]-xaxis[1]/2)/2,(xaxis[1]/2)/2,int(data.shape[0]/2)) 
    y=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 

    data = data.transpose()
    datamiddle = data[64:192,:]    
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    
    plt.axis([ y.min(), y.max(),x.min(), x.max()])
    plt.title(figure_title)

    cs1 = plt.pcolor(y,x,datamiddle,vmin=datamin,vmax=datamax)

    plt.colorbar(cs1, pad = 0.15)

    ax1.set_xlabel(xlabel)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(ylabel_left, color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    middle_index = int(data.shape[0]/2)
    lineout_index = int (middle_index + lineout_position * 20) 
    # Roughly speaking, 1 distance(in normalized units) in x correspond to 20 rows in the data matrix. 256/12.778 = 20
    if (lineout_index >= 64 and lineout_index < 192):
        E_z = data[lineout_index,:]
    else:
        E_z = data[middle_index,:]
     
    ax2.plot(y, E_z, 'r')
    ax2.set_ylabel(ylabel_right, color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    return


def Fx_plot(filename_Ex_XZ,filename_By_XZ,figure_title='',xlabel_down='',xlabel_up='',ylabel='',datamin=-1,datamax=1,lineout_position=0):
    fEx=h5py.File(filename_Ex_XZ,'r')
    fBy=h5py.File(filename_By_XZ,'r')
    positionOf_=filename_Ex_XZ.find('_')
    firstPartOfFilename = filename_Ex_XZ[0:positionOf_]
    dataset_Ex = fEx['/'+firstPartOfFilename]
    data_Ex = dataset_Ex[...]
    positionOf_=filename_By_XZ.find('_')
    firstPartOfFilename = filename_By_XZ[0:positionOf_]
    dataset_By = fBy['/'+firstPartOfFilename]
    data_By = dataset_By[...]
    data = data_By - data_Ex
    xaxis=fEx['/AXIS/AXIS1'][...]
    yaxis=fEx['/AXIS/AXIS2'][...]

    # Only plot the middle(Along x) part of data. Shift the center of x to 0 and decrease the length of x by 1/2 
    x=np.linspace((xaxis[0]-xaxis[1]/2)/2,(xaxis[1]/2)/2,int(data.shape[0]/2)) 
    y=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 

    data = data.transpose()
    datamiddle = data[64:192,:] 
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    
    plt.axis([ y.min(), y.max(),x.min(), x.max()])
    #plt.title(figure_title)

    cs1 = plt.pcolor(y,x,datamiddle,vmin=datamin,vmax=datamax)

    plt.colorbar(cs1, pad = 0.15)

    ax1.set_ylabel(ylabel)
  
    ax1.set_xlabel(xlabel_down, color='b')
    ax1.tick_params('x', colors='b')

    ax2 = ax1.twiny()

    lineout_index = int(lineout_position * 25)
    # Roughly speaking, 1 distance(in normalized units) in xi(ct-z) correspond to 25 columns in the data matrix. 256/10.2415 = 25
    if (lineout_index >= 0 and lineout_index < 256):
        F_x = datamiddle[:,lineout_index]
    else:
        F_x = datamiddle[:,0]
     
    ax2.plot(F_x,x, 'r')
    ax2.set_xlabel(xlabel_up, color='r')
    ax2.tick_params('x', colors='r')

    fig.tight_layout()
    
    return

