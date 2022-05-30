import sys
b= sys.path
sys.path=['/home/jovyan/analysis'] + b
import osiris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from osh5vis import new_fig, osplot
from osh5io import read_h5
from osh5utils import fft
import glob
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
import ipywidgets as widgets
interact_calc=interact_manual.options(manual_name="Make New Input and Run")
import os
from osiris import tajima


def newifile(iname='os-stdin', oname='output.txt', 
             a0=4.0, omega0=36.0, t_flat=0, t_rise=5.6, t_fall=5.6,
            w0= 4.0, ndump=2,  tmax=200.0 ):

    with open(iname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        if 'ndump=' in data[i]:
            data[i] = '    ndump = '+str(ndump)+',\n'
        if 'tmax=' in data[i]:
            data[i] = '    tmax ='+str(tmax)+'\n'    
        if 'a0=' in data[i] and 'omega0' not in data[i]:
            data[i] = '    a0= '+str(a0)+',\n'
        if 'omega0=' in data[i]:
            data[i] = '    omega0 = '+str(omega0)+',\n'
        if 'lon_flat=' in data[i]:
            data[i] = '    lon_flat = '+str(t_flat)+',\n'
        if 'lon_rise=' in data[i]:
            data[i] = '    lon_rise = '+str(t_rise)+',\n'
        if 'lon_fall=' in data[i]:
            data[i] = '    lon_fall = '+str(t_fall)+',\n'
        if 'per_w0=' in data[i]:
            data[i] = '    per_w0 = '+str(2*np.sqrt(a0))+',\n'    

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = 'output'
    print('Running OSIRIS in directory '+dirname+'...')
    osiris.runosiris_2d(rundir=dirname,inputfile=oname, print_out='yes',combine=' ')
   
    
    print('Done')
 
def moving_widget():
    
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='os-stdin', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value='input.txt', description='New Output File:',style=style,layout=layout)
    #velocity
   
    # a0
    d = widgets.FloatText(value=4.0,description='a0:',style=style,layout=layout)
    # frequency 
    e = widgets.BoundedFloatText(value=36.2925, min=0, max=50, description='omega0:',style=style,layout=layout)
    # time 
    f = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_flat:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=5.64152, min=0, max=100, description='t_rise:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=5.64152, min=0, max=100, description='t_fall:',style=style,layout=layout)
    
    w0 = widgets.BoundedFloatText(value=4.0, min=0.0, max=10.0, description='spot size W0:',style=style,layout=layout)
    #xmaxw = widgets.FloatText(value=24, description='xmax:', style=style, layout=layout)
    ndumpw = widgets.IntText(value=3,min = 3, max = 10, description='ndump:', style=style, layout=layout)
    #ppc = widgets.IntText(value=2, min = 1, max = 3, description='Particles per cell:', style=style, layout=layout)
    tmaxw = widgets.FloatText(value=50, description='tmax:', style=style, layout=layout)
    #nx_pw = widgets.IntText(value=2400, description="Number of cells:", style=style, layout=layout)

    im_moving = interact_calc(newifile, iname=a,oname=b,a0=d,omega0=e,t_flat=f, 
                  t_rise=g, w0=w0,t_fall=h, ndump=ndumpw,  tmax=tmaxw)
    im_moving.widget.manual_button.layout.width='250px'
