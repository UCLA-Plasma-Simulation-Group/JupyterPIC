import sys
import glob
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
import ipywidgets as widgets
interact_calc=interact_manual.options(manual_name="Make New Input and Run")
import os
import osiris
from scipy import optimize
import numpy as np
import h5py
import matplotlib.pyplot as plt
    
def newifile_channel(oname='laser_focus_channel_1', Num_core= 1, spot_size=4.0, Num_zr = 1.0, laser_a0 = 0.05, plasma_dens = 17.4, channel_width=4.0, channel_depth = 0.79,  run_osiris=True):
    
    spot_size = spot_size/0.8
    channel_width = channel_width/0.8
    
    tmax = 5 + (np.pi * (spot_size*0.8) **2 / 0.8**2 ) * Num_zr
    plasma_dens = plasma_dens * 4*np.pi**2

    laser_model = 'zpulse'
    if laser_model == 'pgc':
        fname = 'channel_focus_pgc.txt'
        ndump = int(tmax/20.0/0.090909)
    else:
        fname = 'channel_focus.txt'
        ndump = int(tmax/20.0/0.09)
    
    with open(fname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        
        if 'node_number(1:2)' in data[i]:
            data[i] = '  node_number(1:2) = '+str(Num_core)+', 1,\n'
        if 'ndump =' in data[i]:
            data[i] = '  ndump = '+str(ndump)+',\n'
        if 'tmax =' in data[i]:
            data[i] = '  tmax = '+str(tmax)+',\n'
        if 'w0 =' in data[i]:
            data[i] = '  w0 = '+str(spot_size)+', \n'
        if 'per_w0  =' in data[i]:
            data[i] = '  per_w0 = '+str(spot_size)+', \n'
        if 'a0 =' in data[i]:
            data[i] = '  a0 = '+str(laser_a0)+',\n'
        if 'density =' in data[i]:
            data[i] = '  density = '+str(plasma_dens)+',\n'
        if 'math_func_expr =' in data[i]:
            # directly give the expression of the plasma density profile
            data[i] = '  math_func_expr = \"(x1>20.0)*(x1<30.0)*((x1-20.0)/10.0)*(1.0+'+str(channel_depth)+' *(x2*x2)/ '+str(channel_width**2)+') + (x1>=30)*(1.0+'+str(channel_depth)+' *(x2*x2)/'+str(channel_width**2)+')\",\n'

    with open(oname+'.txt','w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+'.txt is written.')
    if run_osiris:
        print('Running OSIRIS in directory '+oname+'...')
        osiris.runosiris_2d(rundir=oname,inputfile=oname+'.txt',print_out='yes',combine='no',np=Num_core)

def laser_in_plasma_channel_widget(run_osiris=True):
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='laser_focus_channel_1', description='New output file:',style=style,layout=layout)
    b = widgets.BoundedIntText(value=1, min=1, max=4, description='Num of processor:',style=style,layout=layout)
    c = widgets.BoundedFloatText(value=1.0, min=1.0, max=3.0, description='Num of $Z_r$:',style=style,layout=layout)
    d = widgets.BoundedFloatText(value=4.0, min=2.0, max=6.0, description='laser_spot_size [um]:',style=style,layout=layout)
    e = widgets.BoundedFloatText(value=0.05, min=0.01, max=0.5, description='laser_a0:',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=0.01, min=0.0, max=1.0, description='plasma_dens [$n_c$]:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=4.0, min=1.0, max=10.0, description='channel_width [um]:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=0.5, min=0.0, max=4.0, description='channel_depth [$n_p$]:',style=style,layout=layout)
    
    im = interact_calc(newifile_channel, oname=a, Num_core=b, Num_zr = c, spot_size=d, laser_a0=e,\
                       plasma_dens=f, channel_width=g, channel_depth=h, run_osiris=fixed(run_osiris));
    im.widget.manual_button.layout.width='250px'
    return a

def newifile_selffocus(oname='laser_focus_selffocus_1', Num_core= 1, spot_size=4.0, laser_a0 = 0.5, plasma_dens = 174.196, run_osiris=True):
    
    spot_size = spot_size/0.8
    
    tmax = 5 + (np.pi * (spot_size*0.8) **2 / 0.8**2 )
    plasma_dens = plasma_dens * 4*np.pi**2
    
    laser_model = 'zpulse'
    
    if laser_model == 'pgc':
        fname = 'self_focus_pgc.txt'
        ndump = int(tmax/20.0/0.090909)
    else:
        fname = 'self_focus.txt'
        ndump = int(tmax/20.0/0.09)
    
    with open(fname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        
        if 'node_number(1:2)' in data[i]:
            data[i] = '  node_number(1:2) = '+str(Num_core)+', 1,\n'
        if 'ndump =' in data[i]:
            data[i] = '  ndump = '+str(ndump)+',\n'
        if 'tmax =' in data[i]:
            data[i] = '  tmax = '+str(tmax)+',\n'
        if 'w0 =' in data[i]:
            data[i] = '  w0 = '+str(spot_size)+', \n'
        if 'per_w0  =' in data[i]:
            data[i] = '  per_w0 = '+str(spot_size)+', \n'
        if 'a0 =' in data[i]:
            data[i] = '  a0 = '+str(laser_a0)+',\n'
        if 'density =' in data[i]:
            data[i] = '  density = '+str(plasma_dens)+',\n'

    with open(oname+'.txt','w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+'.txt is written.')
    if run_osiris:
        print('Running OSIRIS in directory '+oname+'...')
        osiris.runosiris_2d(rundir=oname,inputfile=oname+'.txt',print_out='yes',combine='no',np=4)

def laser_in_plasma_selffocus_widget(run_osiris=True):
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='laser_focus_selffocus_1', description='New output file:',style=style,layout=layout)
    b = widgets.BoundedIntText(value=1, min=1, max=4, description='Num of processor:',style=style,layout=layout)
    c = widgets.BoundedFloatText(value= 4.0, min= 2.0, max=8.0, description='laser_spot_size [um]:',style=style,layout=layout)
    d = widgets.BoundedFloatText(value=0.5, min=0.01, max=1.0, description='laser_a0:',style=style,layout=layout)
    e = widgets.FloatText(value=0.1, min=0.0, max=1.0, description='plasma_dens [$n_c$]:',style=style,layout=layout)
    
    im = interact_calc(newifile_selffocus, oname=a,Num_core=b,spot_size=c,laser_a0=d,plasma_dens=e, run_osiris=fixed(run_osiris));
    im.widget.manual_button.layout.width='250px'
    return a

def plot_data(dirname = 'laser_focus_1', Num = 0):
    
    n0 = 4.4124355e19
    x0 = 0.8
    y0 = 0.8
    E0 = 9.613*1e-10*1e11*n0**0.5 
    laser_model = 'zpulse'
    
    with open(dirname+'.txt') as osdata:
        data = osdata.readlines()
    
    for i in range(len(data)):
        if 'tmax =' in data[i]:
            tmax = float(data[i].split(" ")[-1][:-2])
        if 'per_w0 =' in data[i]:
            w0 = float(data[i].split(" ")[-2][:-1])*0.8
#             print('spot size '+str(w0)+ ' [um]')
            
    index = Num
    
    if laser_model == 'pgc':
        a_data = h5py.File('./' + dirname + '/MS/FLD/a_mod/a_mod-%.6d.h5'% Num, 'r')

        a0 = - a_data['a_mod'][()]*E0

        xmin = a_data['AXIS']['AXIS1'][()][0]
        xmax = a_data['AXIS']['AXIS1'][()][1]
        ymin = a_data['AXIS']['AXIS2'][()][0]
        ymax = a_data['AXIS']['AXIS2'][()][1]
        x = np.linspace(xmin, xmax, a0.shape[1])*x0
        y = np.linspace(ymin, ymax, a0.shape[0])*y0

        I_laser = a0**2
        
    else:
        Ey_data = h5py.File('./' + dirname + '/MS/FLD/e2/e2-%.6d.h5'% Num, 'r')
        Ez_data = h5py.File('./' + dirname + '/MS/FLD/e3/e3-%.6d.h5'% Num, 'r')
        
        Ey = - Ey_data['e2'][()]*E0
        Ez = - Ez_data['e3'][()]*E0

        xmin = Ez_data['AXIS']['AXIS1'][()][0]
        xmax = Ez_data['AXIS']['AXIS1'][()][1]
        ymin = Ez_data['AXIS']['AXIS2'][()][0]
        ymax = Ez_data['AXIS']['AXIS2'][()][1]
        x = np.linspace(xmin, xmax, Ez.shape[1])*x0
        y = np.linspace(ymin, ymax, Ez.shape[0])*y0
        I_laser = Ey**2
        
    i_laser = np.ma.masked_array(I_laser, abs(I_laser) < 1e16);
    I_laser_max = abs(I_laser).max()

    plt.figure(figsize=(12,6))
    
    plt.title('Initial spot size w0 = '+str(w0)+' [um]')
    plt.xlim(0,16+tmax*0.8)
    plt.ylim(-2.5*w0,2.5*w0)
    plt.xlabel('x [um]')
    plt.ylabel('y [um]')
    
    plt.pcolormesh(x,y,i_laser,cmap = 'Blues', vmin = I_laser_max*0.01, vmax = I_laser_max/6.0)
    plt.colorbar()
    
    z = np.linspace(0, 1000, 2000)
    lamda_c = 0.8
    
    z_r = np.pi* w0**2/lamda_c
    r = w0 * np.sqrt(1+(z-16)**2/z_r**2)
    
    plt.plot(z,r, 'r--', label = 'With diffraction')
    plt.plot(z,-r, 'r--')
    plt.plot([0, max(z)], [w0,w0], 'k--', label = 'No diffraction')
    plt.plot([0, max(z)], [-w0,-w0], 'k--')
    plt.legend()
    plt.show() 
    