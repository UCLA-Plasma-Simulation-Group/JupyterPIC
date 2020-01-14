# in the output data folder. For testing purpose
import os
import shutil
import subprocess
from IPython.display import display
import h5py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, FloatSlider, HBox, VBox, interactive_output
from h5_utilities import *
from analysis import *
from scipy.optimize import fsolve


import json
from collections import OrderedDict
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
import ipywidgets as widgets
interact_calc=interact_manual.options(manual_name="Make New Input!")

def deleteAllOutput():
    if(os.path.exists('Beam0001')):
        shutil.rmtree('Beam0001')
    if(os.path.exists('Beam0002')):
        shutil.rmtree('Beam0002')
    if(os.path.exists('Fields')):
        shutil.rmtree('Fields')
    if(os.path.exists('Species0001')):
        shutil.rmtree('Species0001')
    if(os.path.exists('ELOG')):
        shutil.rmtree('ELOG')

def plotSourceProfile():
    rb = 2
    Delta = 0.2
    epsilon = 0.01
    peak = rb**2/((rb+Delta)**2 - rb**2)
    r = [0,rb,rb+epsilon,rb+Delta,rb+Delta+epsilon,3*rb]
    source = [-1,-1,peak,peak,0,0]
    plt.plot(r,source)
    plt.xlabel('$r$')
    plt.ylabel('$-(\\rho - J_z)$')
    #ylim([-1,5])
    plt.yticks(np.arange(-1, 6, 1))
    plt.title('$r_b = 2, \Delta = 0.1$') 
#     plt.rcParams.update({'font.size': 25})
    plt.show()

def makeInput(inputDeckTemplateName,
                 indx,indz,n0,dt,nbeams,time,ndump2D,
                 boxXlength,boxYlength,boxZlength,
                 z_driver,
                 sigma_r_driver,sigma_z_driver,
                 sigma_vx_driver,sigma_vy_driver,
                 gammaE_driver,energySpread_driver,
                 peak_density_driver,
                 piecewise_z1_witness,piecewise_z2_witness,
                 sigma_r_witness,
                 sigma_vx_witness,sigma_vy_witness,
                 gammaE_witness,energySpread_witness,
                 peak_density_witness): 
    
   # if(plasmaDensityProfile == 'piecewise'):
   #     piecewise_density = np.loadtxt(fname = plasmaDataFile, delimiter=",")
   #     piecewise_z = np.loadtxt(fname = zDataFile, delimiter=",")
    
    # If the unit system is SI, then the peak density is the total charge, and all the quantities should be converted into 
    # normalized units
    #if(units == 'SI'):
    #    # Convert everything into SI units first
    #    boxXlength = boxXlength / 1e6; boxYlength = boxYlength / 1e6; boxZlength = boxZlength / 1e6;
    #    z_driver = z_driver / 1e6; z_witness = z_witness/1e6;
    #    sigma_x_driver = sigma_x_driver/1e6; sigma_y_driver = sigma_y_driver/1e6; sigma_z_driver = sigma_z_driver/1e6;
    #    sigma_x_witness = sigma_x_witness/1e6; sigma_y_witness = sigma_y_witness/1e6; sigma_z_witness = sigma_z_witness/1e6;
    #    if(plasmaDensityProfile == 'piecewise'):
    #        piecewise_z = piecewise_z/1e6
    #        
    #    # These peak_densities are actually Q. Convert the units from nC to C
    #    Q_driver = peak_density_driver/1e9;Q_witness = peak_density_witness/1e9;
    #    
    #    # Calculate the peak densities (normalized to plasma density)
    #    peak_density_driver = Q_driver/e/sqrt((2*pi)**3) / sigma_x_driver / sigma_y_driver / sigma_z_driver / (n0 * 1e22)
    #    peak_density_witness = Q_witness/e/sqrt((2*pi)**3) / sigma_x_witness / sigma_y_witness / sigma_z_witness / (n0 * 1e22)
    #    
    #    # Convert all the quantities from SI units to normalized units (except the peak densities, which are ready to use)
    #    wp = sqrt(n0 * 1e22 * e * e / epsilon0 / m)
    #    kp = 1/(c / wp);
    #    
    #    boxXlength = boxXlength * kp; boxYlength = boxYlength * kp; boxZlength = boxZlength * kp;
    #    z_driver = z_driver * kp; z_witness = z_witness * kp;
    #    sigma_x_driver = sigma_x_driver * kp; sigma_y_driver = sigma_y_driver * kp; sigma_z_driver = sigma_z_driver * kp;
    #    sigma_x_witness = sigma_x_witness * kp; sigma_y_witness = sigma_y_witness * kp; sigma_z_witness = sigma_z_witness * kp;
    #    if(plasmaDensityProfile == 'piecewise'):
    #        piecewise_z = piecewise_z * kp
    
    # Get the file object from the template QuickPIC input file

    with open(inputDeckTemplateName) as ftemplate:
        inputDeck = json.load(ftemplate,object_pairs_hook=OrderedDict)

    # Modify the parameters in the QuickPIC input file

    inputDeck['simulation']['indx'] = indx
    inputDeck['simulation']['indy'] = indx
    inputDeck['simulation']['indz'] = indz
    inputDeck['simulation']['n0'] = n0 * 1e16
    
    inputDeck['simulation']['dt'] = dt
    inputDeck['simulation']['nbeams'] = nbeams
    inputDeck['simulation']['time'] = time
    
    inputDeck['simulation']['box']['x'][0] = - boxXlength / 2
    inputDeck['simulation']['box']['x'][1] = boxXlength / 2
    inputDeck['simulation']['box']['y'][0] = - boxYlength / 2
    inputDeck['simulation']['box']['y'][1] = boxYlength / 2
    inputDeck['simulation']['box']['z'][1] = boxZlength
    
    inputDeck['beam'][0]['gamma'] = gammaE_driver
    inputDeck['beam'][0]['peak_density'] = peak_density_driver
    
    inputDeck['beam'][0]['center'][2] = z_driver
    inputDeck['beam'][0]['sigma'] = [sigma_r_driver, sigma_r_driver, sigma_z_driver]
    inputDeck['beam'][0]['sigma_v'] = [sigma_vx_driver,sigma_vy_driver,gammaE_driver * energySpread_driver/100]
    
    inputDeck['beam'][1]['gamma'] = gammaE_witness
    inputDeck['beam'][1]['peak_density'] = peak_density_witness
    
    inputDeck['beam'][1]['piecewise_z'][1] = piecewise_z1_witness
    inputDeck['beam'][1]['piecewise_z'][2] = piecewise_z2_witness
    inputDeck['beam'][1]['piecewise_z'][0] = piecewise_z1_witness-0.01
    inputDeck['beam'][1]['piecewise_z'][3] = piecewise_z2_witness+0.01
    
    inputDeck['beam'][1]['center'][2] = piecewise_z1_witness
    inputDeck['beam'][1]['sigma'] = [sigma_r_witness,sigma_r_witness]
    inputDeck['beam'][1]['sigma_v'] = [sigma_vx_witness,sigma_vy_witness,gammaE_witness * energySpread_witness/100]
    
    #inputDeck['species'][0]['longitudinal_profile'] = plasmaDensityProfile
    #if(plasmaDensityProfile == 'piecewise'):
    #    inputDeck['species'][0]['piecewise_density'] = list(piecewise_density)
    #    inputDeck['species'][0]['piecewise_s'] = list(piecewise_z)
    
    ################# Diagnostic #################
    
    xzSlicePosition = 2 ** (indx - 1) + 1
    yzSlicePosition = 2 ** (indx - 1) + 1
    xySlicePosition = 2 ** (indx - 2) * 3
    # 2D xz slice position
    inputDeck['beam'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['beam'][1]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['field']['diag'][1]['slice'][0][1] = xzSlicePosition
    # 2D yz slice position
    inputDeck['beam'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['beam'][1]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['field']['diag'][1]['slice'][1][1] = yzSlicePosition
    # 2D xy slice position
    inputDeck['beam'][0]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['beam'][1]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['field']['diag'][1]['slice'][2][1] = xySlicePosition
    # 2D dumping frequency
    inputDeck['beam'][0]['diag'][1]['ndump'] = ndump2D
    inputDeck['beam'][0]['diag'][2]['ndump'] = ndump2D
    inputDeck['beam'][1]['diag'][1]['ndump'] = ndump2D
    inputDeck['beam'][1]['diag'][2]['ndump'] = ndump2D
    inputDeck['species'][0]['diag'][1]['ndump'] = ndump2D
    inputDeck['field']['diag'][1]['ndump'] = ndump2D
    # Save the changes to 'qpinput.json'

    with open('qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
        
    print('Lambda (driver) = ',peak_density_driver*sigma_r_driver**2)
    
def makeWidgetsForInput():        
    style = {'description_width': '350px'}
    layout = Layout(width='55%')
    
    inputDeckTemplateNameW = widgets.Text(value='qpinput_profile1.json', description='Template Input File:',style=style,layout=layout)

    #unitsW = widgets.Dropdown(options=['Normalized', 'SI'],value='Normalized', description='Units:',style=style,layout=layout)
    #plasmaDensityProfileW = widgets.Dropdown(options=['uniform', 'piecewise'],value='uniform', description='Longitudinal Plasma Density Profile:',style=style,layout=layout)
    
    #plasmaDataFileW = widgets.Text(value='plasma.txt', description='Plasma Data File:',style=style,layout=layout)
    #zDataFileW = widgets.Text(value='z.txt', description='z Data File (Normalized/$\mu$m):',style=style,layout=layout)
    
    indxW = widgets.IntText(value=8, description='indx (indy):', style=style, layout=layout)
    indzW = widgets.IntText(value=8, description='indz:', style=style, layout=layout)

    n0W = widgets.FloatText(value=4, description='$n_0\;(10^{16}/cm^3)$:', style=style, layout=layout)

    dtW = widgets.IntText(value=10, description='dt:', style=style, layout=layout)
    nbeamsW = widgets.IntText(value=2, description='nbeams:', style=style, layout=layout)
    #nbeamsW = widgets.IntSlider(value=2,min=1,max=2,step=1, description='number of beams:',style=style, layout=layout)
    timeW = widgets.FloatText(value=11, description='time:', style=style, layout=layout)
    ndump2DW = widgets.IntText(value=1, description='dump the data every ? time steps:', style=style, layout=layout)
    
    boxXlengthW = widgets.FloatText(value=12.8, description='boxXlength (Normalized):', style=style, layout=layout)
    boxYlengthW = widgets.FloatText(value=12.8, description='boxYlength (Normalized):', style=style, layout=layout)
    boxZlengthW = widgets.FloatText(value=10.0, description='boxZlength (Normalized):', style=style, layout=layout)
    
    # Driving beam

    z_driverW = widgets.FloatText(value=2.5, description='driver z position (Normalized):', style=style, layout=layout)

    sigma_r_driverW = widgets.FloatText(value=0.5, description='$\sigma_r$(Normalized) (driver):', style=style, layout=layout)
    #sigma_y_driverW = widgets.FloatText(value=0.5, description='$\sigma_y$(Normalized) (driver):', style=style, layout=layout)
    sigma_z_driverW = widgets.FloatText(value=0.4, description='$\sigma_z$(Normalized) (driver):', style=style, layout=layout)
    
    sigma_vx_driverW = widgets.FloatText(value=0, description='$\sigma_{px}$ (driver):', style=style, layout=layout)
    sigma_vy_driverW = widgets.FloatText(value=0, description='$\sigma_{py}$ (driver):', style=style, layout=layout)
    
    gammaE_driverW = widgets.FloatText(value=20000, description='$\gamma$ (driver):', style=style, layout=layout)    
    energySpread_driverW = widgets.FloatText(value=0.0, description='$\Delta \gamma /\gamma$ (%) (driver):', style=style, layout=layout)
    peak_density_driverW = widgets.FloatText(value=0.11, description='$n_{peak}$ (Normalized) (driver):', style=style, layout=layout)

    # Witness beam

    #z_witnessW = widgets.FloatText(value=3.5, description='witness z position (Normalized/$\mu$m):', style=style, layout=layout)
    
    piecewise_z1_witnessW = widgets.FloatText(value=3.5, description='front of witness beam (Normalized):', style=style, layout=layout)
    piecewise_z2_witnessW = widgets.FloatText(value=5.0, description='back of witness beam (Normalized):', style=style, layout=layout)

    sigma_r_witnessW = widgets.FloatText(value=0.5, description='$\sigma_r$(Normalized) (witness):', style=style, layout=layout)
    #sigma_y_witnessW = widgets.FloatText(value=0.5, description='$\sigma_y$(Normalized) (witness):', style=style, layout=layout)
    
    sigma_vx_witnessW = widgets.FloatText(value=0.0, description='$\sigma_{px}$ (witness):', style=style, layout=layout)
    sigma_vy_witnessW = widgets.FloatText(value=0.0, description='$\sigma_{py}$ (witness):', style=style, layout=layout)
    
    gammaE_witnessW = widgets.FloatText(value=20000, description='$\gamma$ (witness):', style=style, layout=layout)    
    energySpread_witnessW = widgets.FloatText(value=0.0, description='$\Delta \gamma /\gamma$ (%) (witness):', style=style, layout=layout)
    peak_density_witnessW = widgets.FloatText(value=0.11, description='$n_{peak}$ (Normalized) (witness):', style=style, layout=layout)
    
    interact_calc(makeInput,inputDeckTemplateName = inputDeckTemplateNameW,
                  indx = indxW,indz=indzW,n0 = n0W,dt=dtW,nbeams=nbeamsW,time = timeW,ndump2D = ndump2DW,
                  boxXlength=boxXlengthW,boxYlength=boxYlengthW,boxZlength=boxZlengthW,
                  z_driver = z_driverW,
                  sigma_r_driver = sigma_r_driverW,sigma_z_driver = sigma_z_driverW,  
                  sigma_vx_driver = sigma_vx_driverW,sigma_vy_driver = sigma_vy_driverW,
                  gammaE_driver = gammaE_driverW,energySpread_driver=energySpread_driverW,
                  peak_density_driver = peak_density_driverW,
                  piecewise_z1_witness=piecewise_z1_witnessW,piecewise_z2_witness=piecewise_z2_witnessW,
                  sigma_r_witness = sigma_r_witnessW,
                  sigma_vx_witness = sigma_vx_witnessW,sigma_vy_witness = sigma_vy_witnessW,
                  gammaE_witness = gammaE_witnessW,energySpread_witness=energySpread_witnessW,
                  peak_density_witness = peak_density_witnessW);

planeToNum = {'xz':'1','yz':'2','xy':'3'}
axisLabel = ('$x\;[c/\omega_p]$','$y\;[c/\omega_p]$',r'$\xi = ct-z\;[c/\omega_p]$',)
# The value to the key is a tuple: (label on the bottom, label on the left)
planeToAxisLabel = {'xz':(axisLabel[2],axisLabel[0]),'yz':(axisLabel[2],axisLabel[1]),'xy':(axisLabel[0],axisLabel[1])}
 
    
#def yujian_action(inputDeckTemplateName,
#                  indx,indy,indz,
#                 n0,boxXlength,boxYlength,boxZlength,z_driver,z_witness,
#                 sigma_x_driver,sigma_y_driver,sigma_z_driver,
#                 sigma_x_witness,sigma_y_witness,sigma_z_witness,
#                 sigma_vx_driver,sigma_vy_driver,
#                 sigma_vx_witness,sigma_vy_witness,
#                 gammaE_driver,gammaE_witness,energySpread_driver,energySpread_witness,
#                 peak_density_driver, peak_density_witness): 
#    
#    # Get the file object from the template QuickPIC input file
#
#    with open(inputDeckTemplateName) as ftemplate:
#        inputDeck = json.load(ftemplate,object_pairs_hook=OrderedDict)
#        
#    # Modify the parameters in the QuickPIC input file
#
#    inputDeck['simulation']['indx'] = indx
#    inputDeck['simulation']['indy'] = indy
#    inputDeck['simulation']['indz'] = indz
#    inputDeck['simulation']['n0'] = n0 * 1e16
#    inputDeck['simulation']['box']['x'][0] = - boxXlength / 2
#    inputDeck['simulation']['box']['x'][1] = boxXlength / 2
#    inputDeck['simulation']['box']['y'][0] = - boxYlength / 2
#    inputDeck['simulation']['box']['y'][1] = boxYlength / 2
#    inputDeck['simulation']['box']['z'][1] = boxZlength
#    
#    inputDeck['beam'][0]['gamma'] = gammaE_driver
#    inputDeck['beam'][1]['gamma'] = gammaE_witness
#    inputDeck['beam'][0]['peak_density'] = peak_density_driver
#    inputDeck['beam'][1]['peak_density'] = peak_density_witness
#    
#    inputDeck['beam'][0]['center'][2] = z_driver
#    inputDeck['beam'][1]['center'][2] = z_witness
#    
#    inputDeck['beam'][0]['sigma'] = [sigma_x_driver,sigma_y_driver,sigma_z_driver]
#    inputDeck['beam'][1]['sigma'] = [sigma_x_witness,sigma_y_witness,sigma_z_witness]
#    inputDeck['beam'][0]['sigma_v'] = [sigma_vx_driver,sigma_vy_driver,gammaE_driver * energySpread_driver/100]
#    inputDeck['beam'][1]['sigma_v'] = [sigma_vx_witness,sigma_vy_witness,gammaE_witness * energySpread_witness/100]
#    
#    ################# Diagnostic #################
#    
#    xzSlicePosition = 2 ** (indx - 1) + 1
#    yzSlicePosition = 2 ** (indy - 1) + 1
#    
#    # 2D xz slice position
#    inputDeck['beam'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
#    inputDeck['beam'][1]['diag'][1]['slice'][0][1] = xzSlicePosition
#    inputDeck['species'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
#    inputDeck['field']['diag'][1]['slice'][0][1] = xzSlicePosition
#    # 2D yz slice position
#    inputDeck['beam'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
#    inputDeck['beam'][1]['diag'][1]['slice'][1][1] = yzSlicePosition
#    inputDeck['species'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
#    inputDeck['field']['diag'][1]['slice'][1][1] = yzSlicePosition
#    
#    # Save the changes to 'qpinput.json'
#
#    with open('qpinput.json','w') as outfile:
#        json.dump(inputDeck,outfile,indent=4)
#
#def makeWidgetsForInput():        
#    style = {'description_width': '350px'}
#    layout = Layout(width='55%')
#
#    inputDeckTemplateNameW = widgets.Text(value='qpinput_nonlinear.json', description='Template Input File:',style=style,layout=layout)
#
#    indxW = widgets.IntText(value=8, description='indx:', style=style, layout=layout)
#    indyW = widgets.IntText(value=8, description='indy:', style=style, layout=layout)
#    indzW = widgets.IntText(value=8, description='indz:', style=style, layout=layout)
#
#
#    n0W = widgets.FloatText(value=1.0, description='n0(10^16/cm^3):', style=style, layout=layout)
#
#    boxXlengthW = widgets.FloatText(value=12, description='boxXlength:', style=style, layout=layout)
#    boxYlengthW = widgets.FloatText(value=12, description='boxYlength:', style=style, layout=layout)
#    boxZlengthW = widgets.FloatText(value=10, description='boxZlength:', style=style, layout=layout)
#
#    z_driverW = widgets.FloatText(value=2.5, description='driver z position:', style=style, layout=layout)
#    z_witnessW = widgets.FloatText(value=6.5, description='witness z position:', style=style, layout=layout)
#
#    sigma_x_driverW = widgets.FloatText(value=0.1, description='$\sigma_x$ (driver):', style=style, layout=layout)
#    sigma_y_driverW = widgets.FloatText(value=0.1, description='$\sigma_y$ (driver):', style=style, layout=layout)
#    sigma_z_driverW = widgets.FloatText(value=0.4, description='$\sigma_z$ (driver):', style=style, layout=layout)
#    sigma_x_witnessW = widgets.FloatText(value=0.1, description='$\sigma_x$ (witness):', style=style, layout=layout)
#    sigma_y_witnessW = widgets.FloatText(value=0.1, description='$\sigma_y$ (witness):', style=style, layout=layout)
#    sigma_z_witnessW = widgets.FloatText(value=0.2, description='$\sigma_z$ (witness):', style=style, layout=layout)
#
#    sigma_vx_driverW = widgets.FloatText(value=3, description='$\sigma_{px}$ (driver):', style=style, layout=layout)
#    sigma_vy_driverW = widgets.FloatText(value=3, description='$\sigma_{py}$ (driver):', style=style, layout=layout)
#    sigma_vx_witnessW = widgets.FloatText(value=3, description='$\sigma_{px}$ (witness):', style=style, layout=layout)
#    sigma_vy_witnessW = widgets.FloatText(value=3, description='$\sigma_{py}$ (witness):', style=style, layout=layout)
#
#    gammaE_driverW = widgets.FloatText(value=20000, description='$\gamma$ (energy) (driver):', style=style, layout=layout)
#    gammaE_witnessW = widgets.FloatText(value=20000, description='$\gamma$ (energy) (witness):', style=style, layout=layout)
#    energySpread_driverW = widgets.FloatText(value=0.5, description='energy spread (%) (driver):', style=style, layout=layout)
#    energySpread_witnessW = widgets.FloatText(value=0.5, description='energy spread (%) (witness):', style=style, layout=layout)
#
#    peak_density_driverW = widgets.FloatText(value=150, description='peak density (driver):', style=style, layout=layout)
#    peak_density_witnessW = widgets.FloatText(value=15, description='peak density (witness):', style=style, layout=layout)
#
#    interact_calc(yujian_action,inputDeckTemplateName = inputDeckTemplateNameW,
#                  indx = indxW,indy=indyW,indz=indzW,
#                  n0 = n0W,boxXlength=boxXlengthW,boxYlength=boxYlengthW,boxZlength=boxZlengthW,
#                  z_driver = z_driverW,z_witness = z_witnessW,
#                  sigma_x_driver = sigma_x_driverW,sigma_y_driver = sigma_y_driverW,sigma_z_driver = sigma_z_driverW, 
#                  sigma_x_witness = sigma_x_witnessW,sigma_y_witness = sigma_y_witnessW,sigma_z_witness = sigma_z_witnessW,
#                  sigma_vx_driver = sigma_vx_driverW,sigma_vy_driver = sigma_vy_driverW,
#                  sigma_vx_witness = sigma_vx_witnessW,sigma_vy_witness = sigma_vy_witnessW,
#                  gammaE_driver = gammaE_driverW,gammaE_witness = gammaE_witnessW,
#                  energySpread_driver = energySpread_driverW, energySpread_witness = energySpread_witnessW,
#                  peak_density_driver = peak_density_driverW, peak_density_witness = peak_density_witnessW);
#
#def yujian_action_linear(inputDeckTemplateName,
#                  indx,indy,indz,
#                 n0,boxXlength,boxYlength,boxZlength,z_driver,
#                 sigma_r_driver,sigma_z_driver,
#                 sigma_vr_driver,
#                 gammaE_driver,energySpread_driver,
#                 peak_density_driver): 
#    
#    # Get the file object from the template QuickPIC input file
#
#    with open(inputDeckTemplateName) as ftemplate:
#        inputDeck = json.load(ftemplate,object_pairs_hook=OrderedDict)
#        
#    # Modify the parameters in the QuickPIC input file
#
#    inputDeck['simulation']['indx'] = indx
#    inputDeck['simulation']['indy'] = indy
#    inputDeck['simulation']['indz'] = indz
#    inputDeck['simulation']['n0'] = n0 * 1e16
#    inputDeck['simulation']['box']['x'][0] = - boxXlength / 2
#    inputDeck['simulation']['box']['x'][1] = boxXlength / 2
#    inputDeck['simulation']['box']['y'][0] = - boxYlength / 2
#    inputDeck['simulation']['box']['y'][1] = boxYlength / 2
#    inputDeck['simulation']['box']['z'][1] = boxZlength
#    
#    inputDeck['beam'][0]['gamma'] = gammaE_driver
#    inputDeck['beam'][0]['peak_density'] = peak_density_driver
#    
#    inputDeck['beam'][0]['center'][2] = z_driver
#    inputDeck['beam'][0]['sigma'] = [sigma_r_driver,sigma_r_driver,sigma_z_driver]
#    inputDeck['beam'][0]['sigma_v'] = [sigma_vr_driver,sigma_vr_driver,gammaE_driver * energySpread_driver/100]
#    
#    ################# Diagnostic #################
#    
#    xzSlicePosition = 2 ** (indx - 1) + 1
#    yzSlicePosition = 2 ** (indy - 1) + 1
#    
#    # 2D xz slice position
#    inputDeck['beam'][0]['diag'][0]['slice'][0][1] = xzSlicePosition
#    inputDeck['species'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
#    inputDeck['field']['diag'][1]['slice'][0][1] = xzSlicePosition
#    # 2D yz slice position
#    inputDeck['beam'][0]['diag'][0]['slice'][1][1] = yzSlicePosition
#    inputDeck['species'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
#    inputDeck['field']['diag'][1]['slice'][1][1] = yzSlicePosition
#    
#    # Save the changes to 'qpinput.json'
#
#    with open('qpinput.json','w') as outfile:
#        json.dump(inputDeck,outfile,indent=4)
#        
#    print('Lambda = ',peak_density_driver*sigma_r_driver**2)
#    
#def makeWidgetsForLinearInput():        
#    style = {'description_width': '350px'}
#    layout = Layout(width='55%')
#
#    inputDeckTemplateNameW = widgets.Text(value='qpinput_linear.json', description='Template Input File:',style=style,layout=layout)
#
#    indxW = widgets.IntText(value=8, description='indx:', style=style, layout=layout)
#    indyW = widgets.IntText(value=8, description='indy:', style=style, layout=layout)
#    indzW = widgets.IntText(value=8, description='indz:', style=style, layout=layout)
#
#
#    n0W = widgets.FloatText(value=1.0, description='n0(10^16/cm^3):', style=style, layout=layout)
#
#    boxXlengthW = widgets.FloatText(value=12.8, description='boxXlength:', style=style, layout=layout)
#    boxYlengthW = widgets.FloatText(value=12.8, description='boxYlength:', style=style, layout=layout)
#    boxZlengthW = widgets.FloatText(value=10, description='boxZlength:', style=style, layout=layout)
#
#    z_driverW = widgets.FloatText(value=2.5, description='driver z position:', style=style, layout=layout)
#   
#
#    sigma_r_driverW = widgets.FloatText(value=0.5, description='sigma_r (driver):', style=style, layout=layout)
#    sigma_z_driverW = widgets.FloatText(value=0.4, description='sigma_z (driver):', style=style, layout=layout)
#    
#    sigma_vr_driverW = widgets.FloatText(value=0, description='sigma_vr (driver):', style=style, layout=layout)
#    
#    gammaE_driverW = widgets.FloatText(value=20000, description='gamma (energy) (driver):', style=style, layout=layout)    
#    energySpread_driverW = widgets.FloatText(value=0, description='energy spread (%) (driver):', style=style, layout=layout)
#    peak_density_driverW = widgets.FloatText(value=0.11, description='peak density (driver):', style=style, layout=layout)
#
#
#    interact_calc(yujian_action_linear,inputDeckTemplateName = inputDeckTemplateNameW,
#                  indx = indxW,indy=indyW,indz=indzW,
#                  n0 = n0W,boxXlength=boxXlengthW,boxYlength=boxYlengthW,boxZlength=boxZlengthW,
#                  z_driver = z_driverW,
#                  sigma_r_driver = sigma_r_driverW,sigma_z_driver = sigma_z_driverW,  
#                  sigma_vr_driver = sigma_vr_driverW,
#                  gammaE_driver = gammaE_driverW,
#                  energySpread_driver = energySpread_driverW,
#                  peak_density_driver = peak_density_driverW);
 
        
def makeplot(rundir):
    isLinear=True
    isSmooth=False
    with open('qpinput.json') as f:
        inputDeck = json.load(f,object_pairs_hook=OrderedDict)
        
    xCellsTotal = 2 ** inputDeck['simulation']['indx'] 
    zCellsTotal = 2 ** inputDeck['simulation']['indz']
    xLengthTotal = inputDeck['simulation']['box']['x'][1] - inputDeck['simulation']['box']['x'][0] 
    zLengthTotal = inputDeck['simulation']['box']['z'][1] - inputDeck['simulation']['box']['z'][0] 
    xCellsPerUnitLength = xCellsTotal/xLengthTotal
    zCellsPerUnitLength = zCellsTotal/zLengthTotal
    
    def plot1_qp(x_position):
        filename=rundir+'/Species0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        figure_title = 'Plasma Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$Plasma Density\;[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$Plasma Density\;[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'viridis'
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 
        data = data.transpose()
        # Only visalize the middle part of the data
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:]  
        l_max = 2.0
        l_min = -10.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title) 
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
            
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        
        #Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * xCellsPerUnitLength)+1
        lineout = da[lineout_index,:]
        ax2.plot(xi, lineout, 'r',label='Simulation')
        if(isLinear == False):
            ax2.set_ylim([l_min,l_max])
        ax1.plot(xi, x_position*np.ones(da.shape[1]), 'w--') # Add a dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        
        boundary = getBubbleBoundary(rundir)
        ax1.plot(boundary[0],boundary[1],'k--',label='bubble boundary')
        
        # Lance's codes
        if(isLinear):
            def rho_para(xi,sigz):
                return np.exp(-xi**2/(2.0*sigz**2))

            def rho_perp(r,sigr):
                return np.exp(-r**2/(2.0*sigr**2))
            
            kpsigr=inputDeck['beam'][0]['sigma'][0]
            kpsigz=inputDeck['beam'][0]['sigma'][2]
            nb=inputDeck['beam'][0]['peak_density']
            R0th=kpsigr**2/2.0*np.exp(kpsigr**2/2.0)*mp.gammainc(0,kpsigr**2/2.0)

            def Z(kpsigz, xi, b, N):
                h=float(b-xi)/N
                k=0.0
                for i in range(1,N):
                    x = xi+i*h
                    if i%2==1:
                        k += 4.0*np.sin(xi-x)*rho_para(x,kpsigz)
                    else:
                        k += 2.0*np.sin(xi-x)*rho_para(x,kpsigz)

                return -(h/3.0)*(rho_para(xi,kpsigz)+np.sin(xi-b)*rho_para(b,kpsigz)+k)
            
            def rhoth(kpxi,kpr):
                if kpxi>0:
                    return nb*rho_perp(kpr,kpsigr)*Z(kpsigz,kpxi,-10.*kpxi,3000)-1.
                else:
                    return nb*rho_perp(kpr,kpsigr)*Z(kpsigz,kpxi,20.*kpxi,3000)-1.
                
            xin=np.linspace(xi[0],xi[-1],50) 
            xi0=inputDeck['beam'][0]['center'][2]
            #ax2.plot(y[y>xi0],ezth(y[y>xi0]-xi0),'k')
            #ax2.plot(y[y>xi0],[ez0(i) for i in y[y>xi0]-xi0],'y--')
            ax2.plot(xin,[rhoth(i,x_position) for i in xin-xi0],'b--',label='Numerical Integration')
            ax2.legend()                     
        
        return

    def plot2_qp(xi_position):
        filename=rundir+'/Species0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 
        figure_title = 'Plasma Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$Plasma Density\;[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$Plasma Density\;[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'viridis'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 2.0
        l_min = -4.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * zCellsPerUnitLength) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r',label='Simulation')
        if(isLinear == False):
            ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(da.shape[0]),x, 'w--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('x', colors='r')
        fig.tight_layout()
        
        # Lance's codes
        if(isLinear):
            def rho_para(xi,sigz):
                return np.exp(-xi**2/(2.0*sigz**2))

            def rho_perp(r,sigr):
                return np.exp(-r**2/(2.0*sigr**2))
            
            kpsigr=inputDeck['beam'][0]['sigma'][0]
            kpsigz=inputDeck['beam'][0]['sigma'][2]
            nb=inputDeck['beam'][0]['peak_density']
            R0th=kpsigr**2/2.0*np.exp(kpsigr**2/2.0)*mp.gammainc(0,kpsigr**2/2.0)

            def Z(kpsigz, xi, b, N):
                h=float(b-xi)/N
                k=0.0
                for i in range(1,N):
                    x = xi+i*h
                    if i%2==1:
                        k += 4.0*np.sin(xi-x)*rho_para(x,kpsigz)
                    else:
                        k += 2.0*np.sin(xi-x)*rho_para(x,kpsigz)

                return -(h/3.0)*(rho_para(xi,kpsigz)+np.sin(xi-b)*rho_para(b,kpsigz)+k)
            
            def rhoth(kpxi,kpr):
                if kpxi>0:
                    return nb*rho_perp(kpr,kpsigr)*Z(kpsigz,kpxi,-10.*kpxi,3000)-1.
                else:
                    return nb*rho_perp(kpr,kpsigr)*Z(kpsigz,kpxi,20.*kpxi,3000)-1.
                
            xn=np.linspace(x[0],x[-1],50) 
            xi0=inputDeck['beam'][0]['center'][2]
            #ax2.plot(y[y>xi0],ezth(y[y>xi0]-xi0),'k')
            #ax2.plot(y[y>xi0],[ez0(i) for i in y[y>xi0]-xi0],'y--')
            ax2.plot([rhoth(xi_position-xi0,i) for i in xn],xn,'b--',label='Numerical Integration')
            ax2.legend()                     
        
        return

    def plot1_qb(x_position):
        filename=rundir+'/Beam0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])  
        # For linear case, we run simulation with only drive beam. For nonlinear case, we also have witness beam
        if(isLinear == False): 
            filename=rundir+'/Beam0002/Charge_slice_0001/charge_slice_xz_00000001.h5'
            f=h5py.File(filename)
            names=list(f.keys())
            dataname='/'+names[1]
            dataset=f[dataname]
            data=data+dataset[...]
            
        figure_title = 'Beam Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Beam Density $[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Beam Density $[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'afmhot'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 0.0
        l_min = -30.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * xCellsPerUnitLength)+1 
        lineout = da[lineout_index,:]
        ax2.plot(xi, lineout, 'r')
#         if(isLinear == False):
#             ax2.set_ylim([l_min,l_max])
        ax1.plot(xi, x_position*np.ones(da.shape[1]), 'b--') # Add a dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return
    
    def plot1_qb_w(x_position):
        filename=rundir+'/Beam0002/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])  
        # For linear case, we run simulation with only drive beam. For nonlinear case, we also have witness beam
        if(isLinear == False): 
            filename=rundir+'/Beam0002/Charge_slice_0001/charge_slice_xz_00000001.h5'
            f=h5py.File(filename)
            names=list(f.keys())
            dataname='/'+names[1]
            dataset=f[dataname]
            data=data+dataset[...]
            
        figure_title = 'Beam Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Beam Density $[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Beam Density $[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'afmhot'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 0.0
        l_min = -30.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * xCellsPerUnitLength)+1 
        lineout = da[lineout_index,:]
        ax2.plot(xi, lineout, 'r')
#         if(isLinear == False):
#             ax2.set_ylim([l_min,l_max])
        ax1.plot(xi, x_position*np.ones(da.shape[1]), 'b--') # Add a dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        
        z1=inputDeck['beam'][1]['piecewise_z'][1]
        z2=inputDeck['beam'][1]['piecewise_z'][2]
        nb_w=inputDeck['beam'][1]['peak_density']
        ax2.plot([z1,z2],[-nb_w,0],'k--')
        ax2.plot([z1-0.01,z1],[0,-nb_w],'k--')
        
        return

    def plot2_qb(xi_position):
        filename=rundir+'/Beam0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        if(isLinear == False):
            filename=rundir+'/Beam0002/Charge_slice_0001/charge_slice_xz_00000001.h5'
            f=h5py.File(filename)
            names=list(f.keys())
            dataname='/'+names[1]
            dataset=f[dataname]
            data=data+dataset[...]
        figure_title = 'Beam Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Beam Density $[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Beam Density $[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'afmhot'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 0.0
        l_min = -30.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * zCellsPerUnitLength) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r')
#         if(isLinear == False):
#             ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(da.shape[0]),x, 'b--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('x', colors='r')
        fig.tight_layout()
        return

    def plot1_ez(x_position):
        filename=rundir+'/Fields/Ez_slice0001/ezslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        figure_title = '$E_z$'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$eE_z/mc\omega_p$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$eE_z/mc\omega_p$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'bwr'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 1.2
        l_min = -1.2
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * xCellsPerUnitLength) 
        lineout = da[lineout_index,:]
        ax2.plot(xi, lineout, 'r',label='Simulation')
        if(isLinear == False):
            ax2.set_ylim([l_min,l_max])
        ax1.plot(xi, x_position*np.ones(da.shape[1]), 'k--') # Add a dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        
        # Lance's codes
        if(isLinear):
            def rho_para(xi,sigz):
                return np.exp(-xi**2/(2.0*sigz**2))

            def rho_perp(r,sigr):
                return np.exp(-r**2/(2.0*sigr**2))

            def R0(kpsigr, b, N):
                h=float(b)/N
                k=0.0
                for i in range(1,N):
                    x = i*h
                    if i%2==1:
                        k += 4.0*x*mp.besselk(0,x)*rho_perp(x,kpsigr)
                    else:
                        k += 2.0*x*mp.besselk(0,x)*rho_perp(x,kpsigr)

                return (h/3.0)*(b*mp.besselk(0,b)*rho_perp(b,kpsigr)+k)
            
            def Rr(kpr,kpsigr, b, N,theta):
                h=float(b)/N
                k=0.0
                for i in range(1,N):
                    x = i*h
                    if sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)) != 0:
                        if i%2==1:
                            k += 4.0*x*mp.besselk(0,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)
                        else:
                            k += 2.0*x*mp.besselk(0,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)

                if sqrt(b**2+kpr**2-2.0*b*kpr*cos(theta)) != 0:
                    return (h/3.0)*(b*mp.besselk(0,sqrt(b**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(b,kpsigr)+k)
                else:
                    return (h/3.0)*k
                                
            def R(kpr,kpsigr,b,N,nth):
                h=float(2.0*np.pi)/nth
                k=0.0
                for i in range(1,nth):
                    theta = i*h
                    if i%2==1:
                        k += 4.0*Rr(kpr,kpsigr,b,N,theta)
                    else:
                        k += 2.0*Rr(kpr,kpsigr,b,N,theta)

                return (h/3.0)*(Rr(kpr,kpsigr,b,N,2.0*np.pi)+Rr(kpr,kpsigr,b,N,0.0)+k)/(2.0*np.pi)

            kpsigr=inputDeck['beam'][0]['sigma'][0]
            kpsigz=inputDeck['beam'][0]['sigma'][2]
            nb=inputDeck['beam'][0]['peak_density']
            R0th=kpsigr**2/2.0*np.exp(kpsigr**2/2.0)*mp.gammainc(0,kpsigr**2/2.0)

            def Zp(kpsigz, xi, b, N):
                h=float(b-xi)/N
                k=0.0
                for i in range(1,N):
                    x = xi+i*h
                    if i%2==1:
                        k += 4.0*np.cos(xi-x)*rho_para(x,kpsigz)
                    else:
                        k += 2.0*np.cos(xi-x)*rho_para(x,kpsigz)

                return -(h/3.0)*(rho_para(xi,kpsigz)+np.cos(xi-b)*rho_para(b,kpsigz)+k)

            def Zpth(kpxi):
                return np.sqrt(2.*np.pi)*kpsigz*np.exp(-kpsigz**2/2.)*np.cos(kpxi)
            
            def ezth(kpxi):
                return nb*Zpth(kpxi)*R0th
            
            def ez0(kpxi):
                #return -nb*Zp(kpsigz,kpxi,10.*kpxi,-1000)*R0(kpsigr,5.*kpsigr,1000)
                #return nb*Zp(kpsigz,kpxi,-100.*kpxi,10000)*R0th
                if kpxi>0:
                    return nb*Zp(kpsigz,kpxi,-5.*kpxi,100)*R0th
                else:
                    return nb*Zp(kpsigz,kpxi,15.*kpxi,15)*R0th

            def ez(kpxi,kpr):
                if kpxi>0:
                    return nb*Zp(kpsigz,kpxi,-5.*kpxi,100)*R(kpr,kpsigr,3.0*kpsigr,15,10)
                else:
                    return nb*Zp(kpsigz,kpxi,15.*kpxi,15)*R(kpr,kpsigr,3.0*kpsigr,15,10)

            xin=np.linspace(xi[0],xi[-1],30) 
            xi0=inputDeck['beam'][0]['center'][2]
            #ax2.plot(y[y>xi0],ezth(y[y>xi0]-xi0),'k')
            #ax2.plot(y[y>xi0],[ez0(i) for i in y[y>xi0]-xi0],'y--')
            #if x_position==0.0:
            #    ax2.plot(xin,ezth(xin-xi0),'k',label='Analytic')
            #    ax2.plot(xin,[ez0(i) for i in xin-xi0],'b--',label='Numerical Integration')
            #    ax2.legend()      
            #else:                    
            #    ax2.plot(xin,[ez(i,x_position) for i in xin-xi0],'b--',label='Numerical Integration')
            #    ax2.legend()                     
                            
        # End of Lance's codes
        else:
            boundary = getBubbleBoundary()
            xi = boundary[0]
            rb = boundary[1]
            #ax1.plot(xi,rb,'k--',label='bubble boundary')
            xiAtRm = getXiAtRm(boundary)
            
            Delta_l = 1.1
            Delta_s = 0.0
            alpha = Delta_l/rb + Delta_s
            beta = ((1+alpha)**2 * np.log((1+alpha)**2))/((1+alpha)**2-1)-1
            
            psi0 = rb**2/4 * (1+beta)
            Ez = Nd(xi,psi0)
            Ez = smooth(Ez)
            ax2.plot(xi,Ez,'r--',label='theory')
            ax2.legend()
        return

    def plot2_ez(xi_position):
        filename=rundir+'/Fields/Ez_slice0001/ezslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        figure_title = '$E_z$'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$eE_z/mc\omega_p$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$eE_z/mc\omega_p$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'bwr'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 1
        l_min = -1
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * zCellsPerUnitLength) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r',label='Simulation')
        if(isLinear == False):
            ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(da.shape[0]),x, 'k--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('x', colors='r')
        fig.tight_layout()
        
        # Lance's codes
        if(isLinear):
            def rho_para(xi,sigz):
                return np.exp(-xi**2/(2.0*sigz**2))

            def rho_perp(r,sigr):
                return np.exp(-r**2/(2.0*sigr**2))

            def R0(kpsigr, b, N):
                h=float(b)/N
                k=0.0
                for i in range(1,N):
                    x = i*h
                    if i%2==1:
                        k += 4.0*x*mp.besselk(0,x)*rho_perp(x,kpsigr)
                    else:
                        k += 2.0*x*mp.besselk(0,x)*rho_perp(x,kpsigr)

                return (h/3.0)*(b*mp.besselk(0,b)*rho_perp(b,kpsigr)+k)
            
            def Rr(kpr,kpsigr, b, N,theta):
                h=float(b)/N
                k=0.0
                for i in range(1,N):
                    x = i*h
                    if sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)) != 0:
                        if i%2==1:
                            k += 4.0*x*mp.besselk(0,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)
                        else:
                            k += 2.0*x*mp.besselk(0,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)

                if sqrt(b**2+kpr**2-2.0*b*kpr*cos(theta)) != 0:
                    return (h/3.0)*(b*mp.besselk(0,sqrt(b**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(b,kpsigr)+k)
                else:
                    return (h/3.0)*k
                                            
            def R(kpr,kpsigr,b,N,nth):
                h=float(2.0*np.pi)/nth
                k=0.0
                for i in range(1,nth):
                    theta = i*h
                    if i%2==1:
                        k += 4.0*Rr(kpr,kpsigr,b,N,theta)
                    else:
                        k += 2.0*Rr(kpr,kpsigr,b,N,theta)

                return (h/3.0)*(Rr(kpr,kpsigr,b,N,2.0*np.pi)+Rr(kpr,kpsigr,b,N,0.0)+k)/(2.0*np.pi)

            kpsigr=inputDeck['beam'][0]['sigma'][0]
            kpsigz=inputDeck['beam'][0]['sigma'][2]
            nb=inputDeck['beam'][0]['peak_density']
            R0th=kpsigr**2/2.0*np.exp(kpsigr**2/2.0)*mp.gammainc(0,kpsigr**2/2.0)

            def Zp(kpsigz, xi, b, N):
                h=float(b-xi)/N
                k=0.0
                for i in range(1,N):
                    x = xi+i*h
                    if i%2==1:
                        k += 4.0*np.cos(xi-x)*rho_para(x,kpsigz)
                    else:
                        k += 2.0*np.cos(xi-x)*rho_para(x,kpsigz)

                return -(h/3.0)*(rho_para(xi,kpsigz)+np.cos(xi-b)*rho_para(b,kpsigz)+k)


            def ezth(kpxi):
                return np.sqrt(2.*np.pi)*nb*kpsigz*np.exp(-kpsigz**2/2.)*R0th*np.cos(kpxi)

            def ez0(kpxi):
                #return -nb*Zp(kpsigz,kpxi,10.*kpxi,-1000)*R0(kpsigr,5.*kpsigr,1000)
                #return nb*Zp(kpsigz,kpxi,-100.*kpxi,10000)*R0th
                if kpxi>0:
                    return nb*Zp(kpsigz,kpxi,-5.*kpxi,100)*R0th
                else:
                    return nb*Zp(kpsigz,kpxi,15.*kpxi,15)*R0th

            def ez(kpxi,kpr):
                if kpxi>0:
                    return nb*Zp(kpsigz,kpxi,-5.*kpxi,100)*R(kpr,kpsigr,3.0*kpsigr,15,10)
                else:
                    return nb*Zp(kpsigz,kpxi,15.*kpxi,15)*R(kpr,kpsigr,3.0*kpsigr,15,10)

            xn=np.linspace(x[0],x[-1],15) 
            xi0=inputDeck['beam'][0]['center'][2]
            #ax2.plot(y[y>xi0],ezth(y[y>xi0]-xi0),'k')
            #ax2.plot(y[y>xi0],[ez0(i) for i in y[y>xi0]-xi0],'y--')
            ax2.plot([ez(xi_position-xi0,i) for i in xn],xn,'b--',label='Numerical Integration')
            ax2.legend()                     
                            
        # End of Lance's codes
        
        
        return

    def plot1_fo(x_position):
        filename=rundir+'/Fields/Ex_slice0001/exslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        filename=rundir+'/Fields/By_slice0001/byslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=-data+dataset[...]
        figure_title = 'Focusing Force'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Focusing Field $[mc\omega_p/e]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Focusing Force $[mc\omega_p/e]$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'jet'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 1.0
        l_min = -1.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * xCellsPerUnitLength)+1 
        lineout = da[lineout_index,:]
        ax2.plot(xi, lineout, 'r',label='Simulation')
        if(isLinear == False):
            ax2.set_ylim([l_min,l_max])
            # plot the theoretical focusing force
            focusing_force_theory = -1/2 * x_position * np.ones(da.shape[1])
            ax2.plot(xi,focusing_force_theory, 'r--',label='F = -1/2 r')
            ax2.legend()
        ax1.plot(xi, x_position*np.ones(da.shape[1]), 'w--') # Add a dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        
        # Lance's codes
        if(isLinear):
            def rho_para(xi,sigz):
                return np.exp(-xi**2/(2.0*sigz**2))

            def rho_perp(r,sigr):
                return np.exp(-r**2/(2.0*sigr**2))

            def Rpr(kpr,kpsigr, b, N,theta):
                h=float(b)/N
                k=0.0
                for i in range(1,N):
                    x = i*h
                    if sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)) != 0:
                        if i%2==1:
                            k += 4.0*x*(x*np.cos(theta)-kpr)/sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta))*mp.besselk(1,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)
                        else:
                            k += 2.0*x*(x*np.cos(theta)-kpr)/sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta))*mp.besselk(1,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)

                if sqrt(b**2+kpr**2-2.0*b*kpr*cos(theta)) != 0:
                    return (h/3.0)*(b*(b*np.cos(theta)-kpr)/sqrt(b**2+kpr**2-2.0*b*kpr*cos(theta))*mp.besselk(1,sqrt(b**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(b,kpsigr)+k)
                else:
                    return (h/3.0)*k
                                            
            def Rp(kpr,kpsigr,b,N,nth):
                h=float(2.0*np.pi)/nth
                k=0.0
                for i in range(1,nth):
                    theta = i*h
                    if i%2==1:
                        k += 4.0*Rpr(kpr,kpsigr,b,N,theta)
                    else:
                        k += 2.0*Rpr(kpr,kpsigr,b,N,theta)

                return (h/3.0)*(Rpr(kpr,kpsigr,b,N,2.0*np.pi)+Rpr(kpr,kpsigr,b,N,0.0)+k)/(2.0*np.pi)

            kpsigr=inputDeck['beam'][0]['sigma'][0]
            kpsigz=inputDeck['beam'][0]['sigma'][2]
            nb=inputDeck['beam'][0]['peak_density']
            R0th=kpsigr**2/2.0*np.exp(kpsigr**2/2.0)*mp.gammainc(0,kpsigr**2/2.0)

            def Z(kpsigz, xi, b, N):
                h=float(b-xi)/N
                k=0.0
                for i in range(1,N):
                    x = xi+i*h
                    if i%2==1:
                        k += 4.0*np.sin(xi-x)*rho_para(x,kpsigz)
                    else:
                        k += 2.0*np.sin(xi-x)*rho_para(x,kpsigz)

                return -(h/3.0)*(rho_para(xi,kpsigz)+np.sin(xi-b)*rho_para(b,kpsigz)+k)
 
            def fperp(kpxi,kpr):
                if kpxi>0:
                    return nb*Z(kpsigz,kpxi,-5.*kpxi,80)*Rp(kpr,kpsigr,3.0*kpsigr,15,10)
                else:
                    return nb*Z(kpsigz,kpxi,10.*kpxi,25)*Rp(kpr,kpsigr,3.0*kpsigr,15,10)
                
            xin=np.linspace(xi[0],xi[-1],30) 
            xi0=inputDeck['beam'][0]['center'][2]
            #ax2.plot(y[y>xi0],ezth(y[y>xi0]-xi0),'k')
            #ax2.plot(y[y>xi0],[ez0(i) for i in y[y>xi0]-xi0],'y--')
            ax2.plot(xin,[fperp(i,x_position) for i in xin-xi0],'b--',label='Numerical Integration')
            ax2.legend()                     
                            
        # End of Lance's codes
        return

    def plot2_fo(xi_position):
        filename=rundir+'/Fields/Ex_slice0001/exslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        filename=rundir+'/Fields/By_slice0001/byslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=-data+dataset[...]
        figure_title = 'Focusing Force'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Focusing Field $[mc\omega_p/e]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Focusing Force $[mc\omega_p/e]$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'jet'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        l_max = 1
        l_min = -1
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        if(isLinear): 
            cs1 = plt.pcolormesh(xi,x,da,cmap=colormap)
        else:
            cs1 = plt.pcolormesh(xi,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * zCellsPerUnitLength) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r',label='Simulation')
        if(isLinear == False):
            ax2.set_xlim([l_min,l_max])
            # plot the 1/2 slope line (theoretical focusing force)
            focusing_force_theory = -1/2 * x
            ax2.plot(focusing_force_theory, x, 'r--',label='F = -1/2 r') 
            ax2.legend() 
        
        ax1.plot(xi_position*np.ones(da.shape[0]),x, 'w--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('x', colors='r')
        fig.tight_layout()
         
        # Lance's codes
        if(isLinear):
            def rho_para(xi,sigz):
                return np.exp(-xi**2/(2.0*sigz**2))

            def rho_perp(r,sigr):
                return np.exp(-r**2/(2.0*sigr**2))

            def Rpr(kpr,kpsigr, b, N,theta):
                h=float(b)/N
                k=0.0
                for i in range(1,N):
                    x = i*h
                    if sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)) != 0:
                        if i%2==1:
                            k += 4.0*x*(x*np.cos(theta)-kpr)/sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta))*mp.besselk(1,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)
                        else:
                            k += 2.0*x*(x*np.cos(theta)-kpr)/sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta))*mp.besselk(1,sqrt(x**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(x,kpsigr)

                if sqrt(b**2+kpr**2-2.0*b*kpr*cos(theta)) != 0:
                    return (h/3.0)*(b*(b*np.cos(theta)-kpr)/sqrt(b**2+kpr**2-2.0*b*kpr*cos(theta))*mp.besselk(1,sqrt(b**2+kpr**2-2.0*x*kpr*cos(theta)))*rho_perp(b,kpsigr)+k)
                else:
                    return (h/3.0)*k
                                            
            def Rp(kpr,kpsigr,b,N,nth):
                h=float(2.0*np.pi)/nth
                k=0.0
                for i in range(1,nth):
                    theta = i*h
                    if i%2==1:
                        k += 4.0*Rpr(kpr,kpsigr,b,N,theta)
                    else:
                        k += 2.0*Rpr(kpr,kpsigr,b,N,theta)

                return (h/3.0)*(Rpr(kpr,kpsigr,b,N,2.0*np.pi)+Rpr(kpr,kpsigr,b,N,0.0)+k)/(2.0*np.pi)

            kpsigr=inputDeck['beam'][0]['sigma'][0]
            kpsigz=inputDeck['beam'][0]['sigma'][2]
            nb=inputDeck['beam'][0]['peak_density']
            R0th=kpsigr**2/2.0*np.exp(kpsigr**2/2.0)*mp.gammainc(0,kpsigr**2/2.0)

            def Z(kpsigz, xi, b, N):
                h=float(b-xi)/N
                k=0.0
                for i in range(1,N):
                    x = xi+i*h
                    if i%2==1:
                        k += 4.0*np.sin(xi-x)*rho_para(x,kpsigz)
                    else:
                        k += 2.0*np.sin(xi-x)*rho_para(x,kpsigz)

                return -(h/3.0)*(rho_para(xi,kpsigz)+np.sin(xi-b)*rho_para(b,kpsigz)+k)
 
            def fperp(kpxi,kpr):
                if kpxi>0:
                    return nb*Z(kpsigz,kpxi,-5.*kpxi,80)*Rp(kpr,kpsigr,2.5*kpsigr,25,13)
                else:
                    return nb*Z(kpsigz,kpxi,10.*kpxi,25)*Rp(kpr,kpsigr,2.5*kpsigr,25,13)
                
            xn=np.linspace(x[0],x[-1],17) 
            xi0=inputDeck['beam'][0]['center'][2]
            #ax2.plot(y[y>xi0],ezth(y[y>xi0]-xi0),'k')
            #ax2.plot(y[y>xi0],[ez0(i) for i in y[y>xi0]-xi0],'y--')
            ax2.plot([fperp(xi_position-xi0,i) for i in xn],xn,'b--',label='Numerical Integration')
            ax2.legend()                     
                            
        # End of Lance's codes
        
        return        
    
    
    
#     if(isLinear):
    i1=interact(plot1_qb,x_position=FloatSlider(min=-3,max=3,step=0.05,continuous_update=False))
    i2=interact(plot1_qb_w,x_position=FloatSlider(min=-3,max=3,step=0.05,continuous_update=False))
    i3=interact(plot1_ez,x_position=FloatSlider(min=-3,max=3,step=0.05,continuous_update=False))
#    i2=interact(plot2_qb,xi_position=FloatSlider(min=0,max=10,step=0.05,value=2.5,continuous_update=False))
#    i3=interact(plot1_qp,x_position=FloatSlider(min=-3,max=3,step=0.05,continuous_update=False))
#    i4=interact(plot2_qp,xi_position=FloatSlider(min=0,max=10,step=0.05,value=4.25,continuous_update=False))
#    i6=interact(plot2_ez,xi_position=FloatSlider(min=0,max=10,step=0.05,value=5.6,continuous_update=False))
#    i7=interact(plot1_fo,x_position=FloatSlider(min=-3,max=3,step=0.05,value=-1.1,continuous_update=False))
#    i8=interact(plot2_fo,xi_position=FloatSlider(min=0,max=10,step=0.05,value=4.1,continuous_update=False))
#     else:    
#         interact(plot1_qp,x_position=(-3,3,0.05))
#         interact(plot2_qp,xi_position=(0,10,0.05))
#         interact(plot1_qb,x_position=(-3,3,0.05))
#         interact(plot2_qb,xi_position=(0,10,0.05))
#         interact(plot1_ez,x_position=(-3,3,0.05))
#         interact(plot2_ez,xi_position=(0,10,0.05))
#         interact(plot1_fo,x_position=(-3,3,0.05))
#         interact(plot2_fo,xi_position=(0,10,0.05))
    return





def smooth(x):
    if(len(x) < 7):
        return x
    x_smooth = np.zeros(len(x))
    x_smooth[0] = x[0]
    x_smooth[1] = (x[0] + x[1] + x[2])/3.0
    x_smooth[2] = (x[0] + x[1] + x[2] + x[3] + x[4])/5.0
    x_smooth[-1] = x[-1]
    x_smooth[-2] = (x[-1] + x[-2] + x[-3])/3.0
    x_smooth[-3] = (x[-1] + x[-2] + x[-3] + x[-4] + x[-5])/5.0
    for i in range(3,len(x) - 3):
        x_smooth[i] = (x[i-3] + x[i-2] + x[i-1] + x[i] + x[i+1] + x[i+2] + x[i+3])/7.0
    return x_smooth

def Nd(x,y):
    if(len(x)!=len(y)):
        print('Length of x and y have to be the same!')
        return
    if(len(x) < 5):
        print('Length of x is too short. Meaningless for numerical derivative!')
        return
    dy = np.zeros(len(x))
    dy[0] = (y[1]-y[0])/(x[1]-x[0])
    dy[1] = (y[2]-y[0])/(x[2]-x[0])
    dy[-1]=(y[-1]-y[-2])/(x[-1]-x[-2])
    dy[-2] = (y[-1]-y[-3])/(x[-1]-x[-3])
    for i in range(2,len(x)-2):
        dy[i] = (-y[i+2] + 8 * y[i+1] - 8 * y[i-1] + y[i-2])/12/((x[i+2]-x[i-2])/4)
    return dy
    
# This function returns [xi_rb;rb_smoothed]
def getBubbleBoundary(rundir):
    filename=rundir+'/Species0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
    f=h5py.File(filename,'r')
    names=list(f.keys())
    dataname='/'+names[1]
    dataset=f[dataname]
    data=dataset[...]
    xaxis=f['/AXIS/AXIS1'][...]
    yaxis=f['/AXIS/AXIS2'][...]

    xiCells = data.shape[0]
    xCells = data.shape[1]
    xMidIndex = int(xCells/2)

    xi=np.linspace(yaxis[0],yaxis[1],xiCells) 
    x=np.linspace(xaxis[0],xaxis[1],xCells) 

    axis = data[:,xMidIndex]

    ionBubbleThreshold = -8e-2

    rb = np.array([])
    xi_rb = np.array([])

    for i in range(xiCells):
        if(axis[i] > ionBubbleThreshold): # The part of axis inside the ion bubble
            xi_rb = np.append(xi_rb,xi[i])
            j = xMidIndex
            while((data[i][j] > ionBubbleThreshold) and (j < xCells)):
                j = j + 1
            rb = np.append(rb,x[j])

    rb_smooth = smooth(rb)
    rb_smooth = smooth(rb_smooth)
    boundary = np.array([xi_rb, rb_smooth])
    return boundary



def getXiAtRm(boundary):
    xi = list(boundary[0])
    rb = list(boundary[1])
    rm_index = rb.index(max(rb))
    xiAtRm = xi[rm_index]
    return xiAtRm