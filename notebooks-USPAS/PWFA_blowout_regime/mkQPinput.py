import os
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, FloatSlider, HBox, VBox, interactive_output

import json
from collections import OrderedDict
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
import ipywidgets as widgets
from math import *
interact_calc=interact_manual.options(manual_name="Make New Input!")

def makeInput(inputDeckTemplateName,
                 indx,indz,n0,
                 boxXlength,boxYlength,boxZlength,
                 z_driver,
                 sigma_x_driver,sigma_y_driver,sigma_z_driver,
                 gammaE_driver,
                 peak_density_driver,
                 ): 
   

    # Get the file object from the template QuickPIC input file

    with open(inputDeckTemplateName) as ftemplate:
        inputDeck = json.load(ftemplate,object_pairs_hook=OrderedDict)

    # Modify the parameters in the QuickPIC input file

    inputDeck['simulation']['indx'] = indx
    inputDeck['simulation']['indy'] = indx
    inputDeck['simulation']['indz'] = indz
    inputDeck['simulation']['n0'] = n0 * 1e16
    
    inputDeck['simulation']['box']['x'][0] = - boxXlength / 2
    inputDeck['simulation']['box']['x'][1] = boxXlength / 2
    inputDeck['simulation']['box']['y'][0] = - boxXlength / 2
    inputDeck['simulation']['box']['y'][1] = boxXlength / 2
    inputDeck['simulation']['box']['z'][1] = boxZlength
    
    inputDeck['beam'][0]['gamma'] = gammaE_driver
    inputDeck['beam'][0]['peak_density'] = peak_density_driver
    
    inputDeck['beam'][0]['center'][2] = z_driver
    inputDeck['beam'][0]['sigma'] = [sigma_x_driver, sigma_y_driver, sigma_z_driver]

    
    ################# Diagnostic #################
    
    xzSlicePosition = 2 ** (indx - 1) + 1
    yzSlicePosition = 2 ** (indx - 1) + 1
    xySlicePosition = 2 ** (indx - 2) * 3
    # 2D xz slice position
    inputDeck['beam'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][0][1] = xzSlicePosition
    inputDeck['field']['diag'][1]['slice'][0][1] = xzSlicePosition
    # 2D yz slice position
    inputDeck['beam'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][1][1] = yzSlicePosition
    inputDeck['field']['diag'][1]['slice'][1][1] = yzSlicePosition
    # 2D xy slice position
    inputDeck['beam'][0]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['species'][0]['diag'][1]['slice'][2][1] = xySlicePosition
    inputDeck['field']['diag'][1]['slice'][2][1] = xySlicePosition

    # Save the changes to 'qpinput.json'

    with open('qpinput.json','w') as outfile:
        json.dump(inputDeck,outfile,indent=4)
    
def makeWidgetsForInput():        
    style = {'description_width': '350px'}
    layout = Layout(width='55%')
    
    inputDeckTemplateNameW = widgets.Text(value='qpinput_template.json', description='Template Input File:',style=style,layout=layout)
    
    indxW = widgets.IntText(value=8, description='indx (indy):', style=style, layout=layout)
    indzW = widgets.IntText(value=8, description='indz:', style=style, layout=layout)


    n0W = widgets.FloatText(value=4, description='$n_0\;(10^{16}/cm^3)$:', style=style, layout=layout)
    
    boxXlengthW = widgets.FloatText(value=10, description='boxXlength:', style=style, layout=layout)
    boxYlengthW = widgets.FloatText(value=10, description='boxYlength:', style=style, layout=layout)
    boxZlengthW = widgets.FloatText(value=8, description='boxZlength:', style=style, layout=layout)
    
    # Driving beam

    z_driverW = widgets.FloatText(value=2, description='driver z position:', style=style, layout=layout)

    sigma_x_driverW = widgets.FloatText(value=0.1, description='$\sigma_x$:', style=style, layout=layout)
    sigma_y_driverW = widgets.FloatText(value=0.1, description='$\sigma_y$:', style=style, layout=layout)
    sigma_z_driverW = widgets.FloatText(value=0.5, description='$\sigma_z$:', style=style, layout=layout)
    
    gammaE_driverW = widgets.FloatText(value=20000, description='$\gamma$:', style=style, layout=layout)    
    peak_density_driverW = widgets.FloatText(value=25, description='$n_{peak}$:', style=style, layout=layout)

    
    interact_calc(makeInput,inputDeckTemplateName = inputDeckTemplateNameW,
                  indx = indxW,indz=indzW,n0 = n0W,
                  boxXlength=boxXlengthW,boxYlength=boxYlengthW,boxZlength=boxZlengthW,
                  z_driver = z_driverW,
                  sigma_x_driver = sigma_x_driverW,sigma_y_driver = sigma_y_driverW,sigma_z_driver = sigma_z_driverW,  
                  gammaE_driver = gammaE_driverW,
                  peak_density_driver = peak_density_driverW
                 );
