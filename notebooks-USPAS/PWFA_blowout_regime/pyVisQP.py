import h5py
import numpy as np
import matplotlib.pyplot as plt

import json
from collections import OrderedDict
from ipywidgets import interact,FloatSlider
import ipywidgets as widgets

# Take numerical differentiation for a 2D numpy array
def NDiff(a,xLength,yLength,Ddir):
    nRows = a.shape[0]
    nCols = a.shape[1]
    dx = xLength / (nCols - 1)
    dy = yLength / (nRows - 1)
    b = a.copy()
    if(Ddir == 'row'):
        b[:,0] = (a[:,1] - a[:,0]) / dx
        b[:,-1] = (a[:,-1] - a[:,-2]) / dx
        b[:,1:-1] = (a[:,2:]-a[:,0:-2])/ (2*dx)
    elif(Ddir == 'column'):
        b[0,:] = (a[1,:] - a[0,:]) / dy
        b[-1,:] = (a[-1,:] - a[-2,:]) / dy
        b[1:-1,:] = (a[2:,:]-a[0:-2,:])/ (2*dy)
    return b

with open('qpinput.json') as f:
    inputDeck = json.load(f,object_pairs_hook=OrderedDict)

    xCellsTotal = 2 ** inputDeck['simulation']['indx'] - 1
    zCellsTotal = 2 ** inputDeck['simulation']['indz'] - 1
    xMax = inputDeck['simulation']['box']['x'][1]
    xMin = inputDeck['simulation']['box']['x'][0]
    xLengthTotal = xMax - xMin
    
    zMax = inputDeck['simulation']['box']['z'][1]
    zMin = inputDeck['simulation']['box']['z'][0]
    zLengthTotal = zMax - zMin

    xCellsPerUnitLength = xCellsTotal/xLengthTotal
    zCellsPerUnitLength = zCellsTotal/zLengthTotal

# Show_theory = 'focus'
# DiffDir = 'r' or 'xi'

def makeplot(fileNameList,scaleList = [1],LineoutDir = None,Show_theory = None,DiffDir = None,specify_title = ''):
    
    # This is the first filename
    filename = fileNameList[0]
    # Depending on what kind of data we are plotting, the best range of the colorbar and lineout is different
    
    if('Species' in filename):
        colorBarDefaultRange = [-5,0]
        colorBarTotalRange = [-10,0]
        lineoutAxisRange = [-10,0]
    elif('Beam' in filename):
        colorBarDefaultRange = [-10,0]
        colorBarTotalRange = [-50,0]
        lineoutAxisRange = [-100,0]
    elif('Fields' in filename):
        colorBarDefaultRange = [-1,1]
        colorBarTotalRange = [-5,5]
        lineoutAxisRange = [-2,2]
    
    # Determine the range of the lineout, depending on the direction of the lineout
    lineoutRange = [0,0]
    if(LineoutDir == 'transverse'):
        lineoutRange = [zMin,zMax]
    elif(LineoutDir == 'longitudinal'):
        lineoutRange = [xMin /2 ,xMax /2]
        
    for i in range(len(fileNameList)):
        f=h5py.File(fileNameList[i],'r')
        k=list(f.keys()) # k = ['AXIS', 'charge_slice_xz']
        DATASET = f[k[1]]
        if(i == 0):
            data = np.array(DATASET) * scaleList[0]
        else:
            data += np.array(DATASET) * scaleList[i]

    AXIS = f[k[0]] # AXIS is a group, which contains two datasets: AXIS1 and AXIS2
     

    LONG_NAME = DATASET.attrs['LONG_NAME']
    UNITS = DATASET.attrs['UNITS']

    title = LONG_NAME[0].decode('UTF-8')
    unit = UNITS[0].decode('UTF-8')

    figure_title = title + ' [$' + unit + '$]' 
    if(specify_title != ''):
        figure_title = specify_title

    #### Read the axis labels and the corresponding units

    AXIS1 = AXIS['AXIS1']
    AXIS2 = AXIS['AXIS2']

    LONG_NAME1 = AXIS1.attrs['LONG_NAME']
    UNITS1 = AXIS1.attrs['UNITS']

    LONG_NAME2 = AXIS2.attrs['LONG_NAME']
    UNITS2 = AXIS2.attrs['UNITS']

    axisLabel1 = LONG_NAME1[0].decode('UTF-8')
    unit1 = UNITS1[0].decode('UTF-8')

    axisLabel2 = LONG_NAME2[0].decode('UTF-8')
    unit2 = UNITS2[0].decode('UTF-8')

    label_bottom = '$'+axisLabel2+'$' + '  $[' + unit2 + ']$' 
    label_left = '$'+axisLabel1+'$' + '  $[' + unit1 + ']$' 

    axis = list(AXIS) # axis = ['AXIS1', 'AXIS2']
    

    xRange=list(f['AXIS/AXIS1'])
    xiRange=list(f['AXIS/AXIS2'])
    
    x=np.linspace(xRange[0],xRange[1],data.shape[1])
    xi=np.linspace(xiRange[0],xiRange[1],data.shape[0]) 
    
    ##### If we need to take a derivative

    if(DiffDir == 'xi'):
        data = NDiff(data,xRange[1] - xRange[0],xiRange[1] - xiRange[0],Ddir = 'column')
    elif(DiffDir == 'r'):
        data = NDiff(data,xRange[1] - xRange[0],xiRange[1] - xiRange[0],Ddir = 'row')

    #####
    
    dataT = data.transpose()
    
    colormap = 'viridis'
    
    def plot(colorBarRange,lineout_position):  

        fig, ax1 = plt.subplots(figsize=(8,5))
        # Zoom in / zoom out the plot
        ax1.axis([ xi.min(), xi.max(),x.min()/2, x.max()/2])
        ###

        ax1.set_title(figure_title)
        

        cs1 = ax1.pcolormesh(xi,x,dataT,vmin=colorBarRange[0],vmax=colorBarRange[1],cmap=colormap)
       
        fig.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(label_bottom)

        #Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(label_left, color='k')
        ax1.tick_params('y', colors='k')
        
        if(LineoutDir == 'longitudinal'):
            ax2 = ax1.twinx()
            middle_index = int(dataT.shape[0]/2)
            lineout_index = int (middle_index + lineout_position * xCellsPerUnitLength)
            lineout = dataT[lineout_index,:]
            ax2.plot(xi, lineout, 'r')
            
            if(Show_theory == 'focus'):
                # plot the 1/2 slope line (theoretical focusing force)
                focusing_force_theory = -1/2 * lineout_position * np.ones(dataT.shape[1])
                ax2.plot(xi,focusing_force_theory, 'r--',label='F = -1/2 r')
                ax2.legend()
            
            
            ax2.set_ylim(lineoutAxisRange)
            ax1.plot(xi, lineout_position*np.ones(dataT.shape[1]), 'b--') # Add a dashed line at the lineout position
            ax2.tick_params('y', colors='r')
        elif(LineoutDir == 'transverse'):
            ax2 = ax1.twiny()
            lineout_index = int (lineout_position * zCellsPerUnitLength) 
            lineout = dataT[:,lineout_index]
            ax2.plot(lineout, x, 'r')
            
            if(Show_theory == 'focus'):
                # plot the 1/2 slope line (theoretical focusing force)
                focusing_force_theory = -1/2 * x
                ax2.plot(focusing_force_theory, x, 'r--',label='F = -1/2 r') 
                ax2.legend()
            
            ax2.set_xlim(lineoutAxisRange)
            ax1.plot(lineout_position*np.ones(dataT.shape[0]),x, 'b--')
            ax2.tick_params('x', colors='r')
      
        fig.tight_layout()
     
        return
   
    i1=interact(plot,
                colorBarRange = widgets.FloatRangeSlider(value=colorBarDefaultRange,min=colorBarTotalRange[0],max=colorBarTotalRange[1],step=0.1,description='Colorbar:',continuous_update=False),
                lineout_position = FloatSlider(min=lineoutRange[0],max=lineoutRange[1],step=0.05,description='lineout position:',continuous_update=False)
               )
    return
