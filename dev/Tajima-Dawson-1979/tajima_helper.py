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

def plot_maxgamma_t(simdir):
    
    maxg, time = [], []
    for f in sorted(glob.glob(simdir + '/MS/PHA/p1x1/electrons/*.h5')):
        data = read_h5(f)
        ind = np.nonzero(data)
        if len(ind[0]==0):
            maxg.append(np.sqrt(1+(data.axes[0][max(ind[0])])**2))
            time.append(data.run_attrs['TIME'])
    print('max gamma = ', max(maxg))
    
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
 

# similar to newfile, but nx_p is set automatically by omega0.
#
#
def newifile2(iname='case0.txt', oname='case1.txt', uth=1e-6, 
             a0=1.0, omega0=2.0, t_flat=3.14, t_rise=0, t_fall=0,
            xmax=102.4, ndump=1, ppc=10, tmax=200.0 ):

    with open(iname) as osdata:
        data = osdata.readlines()
    delta_x = 0.15/omega0
    nx_p = int(xmax/delta_x)
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
    
def laser_envelope(trise,tflat,tfall):
    def osiris_env(tau):
        return(10*tau*tau*tau-15*tau*tau*tau*tau+6*tau*tau*tau*tau*tau)
    npoints=201
    total_range=(trise+tflat+tfall)
    dt=total_range/(npoints)
    xaxis=np.arange(0,total_range,dt)
    yaxis=np.zeros(npoints)
    yaxis_gaussian=np.zeros(npoints)
    x_midpoint=total_range/2.0
    
    for i in range(0,npoints):
        if (xaxis[i]<trise):
            yaxis[i]=osiris_env(xaxis[i]/trise)
        elif (xaxis[i]<(trise+tflat)):
            yaxis[i]=1.0
        else:
            yaxis[i]=osiris_env(np.abs(total_range-xaxis[i])/tfall)
        yaxis_gaussian[i]=np.exp(-(xaxis[i]-x_midpoint)**2/(trise/2*trise/2))
    plt.figure()
    if len(xaxis) == len(yaxis):
        plt.plot(xaxis,yaxis,label='OSIRIS Shape') 
        plt.plot(xaxis,yaxis_gaussian,label='Gaussian, $\sigma=t_{rise}/2$')
    else:
        plt.plot(xaxis,np.append(yaxis, 0),label='OSIRIS Shape') 
        plt.plot(xaxis,np.append(yaxis_gaussian, 0),label='Gaussian, $\sigma=t_{rise}/2$')
    plt.legend()
    plt.show()

def tajima_widget():
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
    
def tajima_moving_widget():
    
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='casea-moving-24.txt', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value='case1-moving.txt', description='New Output File:',style=style,layout=layout)
    c = widgets.BoundedFloatText(value=0.0001, min=0.0, max=2.0, description='v_e/c:',style=style,layout=layout)
    d = widgets.FloatText(value=1.0,description='a0:',style=style,layout=layout)
    e = widgets.BoundedFloatText(value=10.0, min=0, max=30, description='omega0:',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=3.14, min=0, max=100, description='t_flat:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_rise:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_fall:',style=style,layout=layout)
    xmaxw = widgets.FloatText(value=24, description='xmax:', style=style, layout=layout)
    ndumpw = widgets.IntText(value=2, description='ndump:', style=style, layout=layout)
    ppc = widgets.IntText(value=10, description='Particles per cell:', style=style, layout=layout)
    tmaxw = widgets.FloatText(value=200, description='tmax:', style=style, layout=layout)

    im_moving = interact_calc(newifile2, iname=a,oname=b,uth=c,a0=d,omega0=e,t_flat=f, 
                  t_rise=g, t_fall=h, xmax=xmaxw, ndump=ndumpw, ppc=ppc, tmax=tmaxw);
    im_moving.widget.manual_button.layout.width='250px'

def xt_and_energy_plot(rundir, field='e2'):
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)

    xlim = [hdf5_data.axes[0].axis_min, hdf5_data.axes[0].axis_max]
    tlim = [hdf5_data.axes[1].axis_min, hdf5_data.axes[1].axis_max]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,5), gridspec_kw={'width_ratios': [1, 3]})
    fig.subplots_adjust(wspace=0.05)

    #This calculates the energy as the electric field squared, summed over x at each time step.
    print(hdf5_data.axes[0].axis_min)
    print(hdf5_data.axes[0].axis_max)
    print(hdf5_data.data.shape)
    dx = (hdf5_data.axes[0].axis_max-hdf5_data.axes[0].axis_min)/hdf5_data.data.shape[1]
    energy = 0.5 * np.sum(hdf5_data.data * hdf5_data.data, axis=1)*dx
    

    axs[0].set_xlabel('Energy [$mc^{2}n_{0}c \omega_{p}^{-1}$]')
    axs[0].set_ylabel('Time [$\omega_p^{-1}$]')
    axs[0].set_ylim(tlim[0],tlim[1]) 
    axs[0].plot(energy, np.linspace(0, tlim[1], len(energy)))

    extent_stuff = [hdf5_data.axes[0].axis_min, hdf5_data.axes[0].axis_max, hdf5_data.axes[1].axis_min,
                        hdf5_data.axes[1].axis_max]
    plt.imshow(hdf5_data.data, extent=extent_stuff, aspect='auto',origin='lower')
    cbar = plt.colorbar()
    cbar.set_label('$E_{' + field[1] +'} [m_e c \omega_{p} e^{-1}]$')

    if field == 'e2':
        field_name = 'Laser'
    if field == 'e1':
        field_name = 'Wake'
    fig.suptitle(field_name + ' Electric Field', fontsize=16)
    plt.xlabel('Position [$c/ \omega_{p}$]')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(tlim[0],tlim[1])  

    fig.show() 

def yujian_action( iname='qpinput.json',oname='qp_new.json', indx=8, indy=8  ):    
    
    with open(iname) as osdata:
        data = osdata.readlines()
        
    for i in range(len(data)):
        if 'indx' in data[i]:
            data[i] = ' \" indx \" :' + str(indx)+' , \n'
        if 'indy' in data[i]:
            data[i] = ' \" indy \" :' + str(indy)+' , \n'

  
    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running QuickPIC in directory '+dirname+'...')
    print('Done')

def yujian_widget():
    
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='qpinput.json', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value='qp_new.json', description='New Output File:',style=style,layout=layout)
    
    indy = widgets.IntText(value=8, description='indy:', style=style, layout=layout)
    indx = widgets.IntSlider(value=8,min=0,max=20,step=1, description='indx:',orientation='horizontal', style=style, layout=layout)

    im_qpic = interact_calc(yujian_action, iname=a, oname = b, indx = indx, indy = indy);
    im_qpic.widget.manual_button.layout.width='250px'
    
    