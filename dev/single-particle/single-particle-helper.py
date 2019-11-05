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
from h5_utilities import *
from analysis import *
import matplotlib.colors as colors

def newifile(iname='case0.txt', oname='case1.txt', uth=1e-6, 
             a0=1.0, omega0=2.0, t_flat=3.14, t_rise=0, t_fall=0,
            nx_p=1024, xmax=102.4, ndump=1, ppc=10, tmax=200.0 ):

    with open(iname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        if 'uth(1:3)' in data[i]:
            data[i] = 'uth(1:3) = '+str(uth)+' , '+str(uth)+' , '+str(uth)+',\n'
        if 'uth_bnd(1:3,1,1)' in data[i]:
            data[i] = 'uth_bnd(1:3,1,1) = '+str(uth)+' , '+str(uth)+' , '+str(uth)+',\n'
        if 'uth_bnd(1:3,2,1)' in data[i]:
            data[i] = 'uth_bnd(1:3,2,1) = '+str(uth)+' , '+str(uth)+' , '+str(uth)+',\n'
        if 'a0 =' in data[i] and 'omega0' not in data[i]:
            data[i] = 'a0 = '+str(a0)+',\n'
        if 'omega0 =' in data[i]:
            data[i] = 'omega0 = '+str(omega0)+',\n'
        if 't_flat =' in data[i]:
            data[i] = 't_flat = '+str(t_flat)+',\n'
        if 't_rise =' in data[i]:
            data[i] = 't_rise = '+str(t_rise)+',\n'
        if 't_fall =' in data[i]:
            data[i] = 't_fall = '+str(t_fall)+',\n'
        if 'nx_p(1:1) =' in data[i]:
            data[i] = 'nx_p(1:1) = '+str(nx_p)+',\n'
        if 'dt =' in data[i]:
            dt = (0.98 * xmax / nx_p)
            data[i] = 'dt = '+str(dt)+'e0'+',\n'
        if 'xmax(1:1) =' in data[i] and 'ps_xmax(1:1)' not in data[i]:
            data[i] = 'xmax(1:1) = '+str(xmax)+'e0,\n'
        if 'ps_xmax(1:1) =' in data[i]:
            data[i] = 'ps_xmax(1:1) = '+str(xmax)+',\n'
        if 'x(1:2,1) =' in data[i] and 'fx(1:2,1)' not in data[i]:
            data[i] = 'x(1:2,1) = 0.0, '+str(xmax)+',\n'
        if 'ndump =' in data[i]:
            data[i] = 'ndump = '+str(ndump)+',\n'
        if 'num_par_x(1:1) =' in data[i]:
            data[i] = 'num_par_x(1:1) = '+str(ppc)+',\n'
        if 'lon_start =' in data[i]:
            data[i] = 'lon_start = '+str(xmax-0.5)+'\n'
# This line is for moving window only
        if ('x(1:4,1)' in data[i]) and not ('fx(1:4,1)' in data[i]):
            data[i]= 'x(1:4,1) = 0.0, '+str(xmax)+' , '+str(xmax+0.001)+', 1500.0,'+'\n'
        if 'tmax =' in data[i]:
            data[i] = 'tmax ='+str(tmax)+'\n'

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running OSIRIS in directory '+dirname+'...')
    osiris.runosiris(rundir=dirname,inputfile=oname,print_out='yes')
    plot_maxgamma_t(dirname)
    print('Done')

def single_part_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='casec-fixed.txt', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value='case1.txt', description='New Output File:',style=style,layout=layout)
    c = widgets.BoundedFloatText(value=0.2, min=0.0, max=2.0, description='v_e/c:',style=style,layout=layout)
    d = widgets.FloatText(value=1.0,description='a0:',style=style,layout=layout)
    e = widgets.BoundedFloatText(value=2.3, min=0, max=9.5, description='omega0:',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=3.14, min=0, max=100, description='Lt:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_rise:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_fall:',style=style,layout=layout)
    nx_pw = widgets.IntText(value=1024, description='nx_p:', style=style, layout=layout)
    xmaxw = widgets.FloatText(value=102.4, description='xmax:', style=style, layout=layout)
    ndumpw = widgets.IntText(value=1, description='ndump:', style=style, layout=layout)
    ppc = widgets.IntText(value=10, description='Particles per cell:', style=style, layout=layout)
    print('d='+repr(d))
    im = interact_calc(newifile, iname=a,oname=b,uth=c,a0=d,omega0=e,t_flat=f, 
                  t_rise=g, t_fall=h, nx_p=nx_pw, xmax=xmaxw, ndump=ndumpw, ppc=ppc);
    im.widget.manual_button.layout.width='250px'