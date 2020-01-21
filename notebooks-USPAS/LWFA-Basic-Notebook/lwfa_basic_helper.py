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
    e = widgets.BoundedFloatText(value=2.3, min=0, max=50, description='omega0:',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=3.14, min=0, max=100, description='Lt:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_rise:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_fall:',style=style,layout=layout)
    nx_pw = widgets.IntText(value=1024, description='nx_p:', style=style, layout=layout)
    xmaxw = widgets.FloatText(value=102.4, description='xmax:', style=style, layout=layout)
    ndumpw = widgets.IntText(value=1, description='ndump:', style=style, layout=layout)
    ppc = widgets.IntText(value=10, description='Particles per cell:', style=style, layout=layout)

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
    e = widgets.BoundedFloatText(value=10.0, min=0, max=50, description='omega0:',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=3.14, min=0, max=100, description='t_flat:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_rise:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_fall:',style=style,layout=layout)
    xmaxw = widgets.FloatText(value=24, description='xmax:', style=style, layout=layout)
    ndumpw = widgets.IntText(value=2, description='ndump:', style=style, layout=layout)
    ppc = widgets.IntText(value=10, description='Particles per cell:', style=style, layout=layout)
    tmaxw = widgets.FloatText(value=200, description='tmax:', style=style, layout=layout)
    nx_pw = widgets.IntText(value=2400, description="Number of cells:", style=style, layout=layout)

    im_moving = interact_calc(newifile, iname=a,oname=b,uth=c,a0=d,omega0=e,t_flat=f, 
                  t_rise=g, t_fall=h, nx_p=nx_pw, xmax=xmaxw, ndump=ndumpw, ppc=ppc, tmax=tmaxw);
    im_moving.widget.manual_button.layout.width='250px'


from ipywidgets import interact, fixed
from h5_utilities import *
from analysis import *
from scipy.optimize import fsolve

from scipy import special

import osh5io
import osh5def
import osh5vis

import osh5utils

import matplotlib.colors as colors

def lwfa_density_plot(rundir):
    import os

    def make_plot(rundir,file_no):

        my_path=os.getcwd()
        working_dir=my_path+'/'+rundir
        
        efield_dir=working_dir+'/MS/FLD/e1/'
        laser_dir = working_dir+'/MS/FLD/e2/'
        eden_dir = working_dir + '/MS/DENSITY/electrons/charge/'
        phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
        p2x1_dir=working_dir+'/MS/PHA/p2x1/electrons/'

        efield_prefix='e1-'
        laser_prefix='e2-'
        phase_prefix='p1x1-electrons-'
        p2x1_prefix='p2x1-electrons-'
        eden_prefix='charge-electrons-'
        
        filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        
        fig = plt.figure(figsize=(12,5) )
        phase_space=np.abs(osh5io.read_h5(filename1))
        time=phase_space.run_attrs['TIME'][0]
        fig.suptitle('Time = '+repr(time)+'$\omega_p^{-1}$',fontsize=18)

        filename2=eden_dir+eden_prefix+repr(file_no).zfill(6)+'.h5'
        filename3=efield_dir+efield_prefix+repr(file_no).zfill(6)+'.h5'

        eden=osh5io.read_h5(filename2)
        ex = osh5io.read_h5(filename3)
        psi = osh5io.read_h5(filename3) 

        den_plot = plt.subplot(121)
        osh5vis.osplot(eden,title='Electron Density')
        
        ex_plot = plt.subplot(122)

        for i in range(psi.shape[0]-2,-1,-1):
            psi[i]=psi[i+1] + psi.axes[0].increment*psi[i]
        

        osh5vis.osplot(psi,title='Wake $\psi$ ',ylabel='$\psi [m_e c^2/e]$')
        
        second_x = plt.twinx()
        second_x.plot(ex.axes[0], ex, 'g', linestyle='-.')
        second_x.set_ylabel('$E_{z}$', color='g')
        second_x.tick_params(axis='y', labelcolor='g')
        

    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
    files=sorted(os.listdir(phase_space_dir))
    start=files[1].find('p1x1-electrons')+16
    end=files[1].find('.')
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(make_plot,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)

def lwfa_laser_plot(rundir):

    import os
    
    def make_plot(rundir,file_no):

        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        efield_dir=working_dir+'/MS/FLD/e1/'
        laser_dir = working_dir+'/MS/FLD/e2/'
        eden_dir = working_dir + '/MS/DENSITY/electrons/charge/'
        phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
        p2x1_dir=working_dir+'/MS/PHA/p2x1/electrons/'

        efield_prefix='e1-'
        laser_prefix='e2-'
        phase_prefix='p1x1-electrons-'
        p2x1_prefix='p2x1-electrons-'
        eden_prefix='charge-electrons-'
        
        filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        
        fig = plt.figure(figsize=(12,5) )
        phase_space=np.abs(osh5io.read_h5(filename1))
        time=phase_space.run_attrs['TIME'][0]
        fig.suptitle('Time = '+repr(time)+'$\omega_p^{-1}$',fontsize=18)

        filename4=laser_dir+laser_prefix+repr(file_no).zfill(6)+'.h5'

        ey = osh5io.read_h5(filename4)


        ey_plot = plt.subplot(121)


        osh5vis.osplot(ey,title='Laser Electric Field')

        ey_plot_k = plt.subplot(122)


        osh5vis.osplot(np.abs(osh5utils.fft(ey)), xlim=[0, 20],linestyle='-',title='k spectrum')

    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
    files=sorted(os.listdir(phase_space_dir))
    start=files[1].find('p1x1-electrons')+16
    end=files[1].find('.')
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(make_plot,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)
    

def lwfa_phase_space(rundir):

    import os

    def make_plot(rundir,file_no):

        my_path=os.getcwd()
        working_dir=my_path+'/'+rundir
        efield_dir=working_dir+'/MS/FLD/e1/'
        phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
        p2x1_dir=working_dir+'/MS/PHA/p2x1/electrons/'

        efield_prefix='e1-'
        phase_prefix='p1x1-electrons-'
        p2x1_prefix='p2x1-electrons-'
        fig = plt.figure(figsize=(12,5) )

        filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        filename3=efield_dir+efield_prefix+repr(file_no).zfill(6)+'.h5'
        filename5=p2x1_dir+p2x1_prefix+repr(file_no).zfill(6)+'.h5'

        phase_space=np.abs(osh5io.read_h5(filename1))
        ex = osh5io.read_h5(filename3)
        p2x1=np.abs(osh5io.read_h5(filename5))

        phase_plot=plt.subplot(121 )
        title=phase_space.data_attrs['LONG_NAME']
        time=phase_space.run_attrs['TIME'][0]

        fig.suptitle('Time = '+repr(time)+'$\omega_p^{-1}$',fontsize=18)
        ext_stuff=[phase_space.axes[1].min,phase_space.axes[1].max,phase_space.axes[0].min,phase_space.axes[0].max]
        data_max=max(np.abs(np.amax(phase_space)),100)
        phase_contour=plt.contourf(np.abs(phase_space+0.000000001),
                    levels=[0.00001*data_max,0.0001*data_max,0.001*data_max,0.01*data_max,0.05*data_max,0.1*data_max,0.2*data_max,0.5*data_max],
                    extent=ext_stuff,cmap='Spectral',vmin=1e-5*data_max,vmax=1.5*data_max,
                    norm=colors.LogNorm(vmin=0.00001*data_max,vmax=1.5*data_max))
        phase_plot.set_title('P1X1 Phase Space')
        phase_plot.set_xlabel('Position [$c / \omega_{p}$]')
        phase_plot.set_ylabel('Proper Velocity $\gamma v_1$ [ c ]')
        second_x = plt.twinx()
        second_x.plot(ex.axes[0],ex,'g',linestyle='-.')
        second_x.tick_params(axis='y', labelcolor='g')

        plt.colorbar(phase_contour)


        p2x1_plot=plt.subplot(122)
        title=p2x1.data_attrs['LONG_NAME']
        time=p2x1.run_attrs['TIME'][0]
        ext_stuff=[p2x1.axes[1].min,p2x1.axes[1].max,p2x1.axes[0].min,p2x1.axes[0].max]
        p2x1_contour=plt.contourf(np.abs(p2x1+0.000000001),levels=[0.00001,0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1,10,100,500],extent=ext_stuff,cmap='Spectral',vmin=1e-5,vmax=3000,
                    norm=colors.LogNorm(vmin=0.0001,vmax=3000))
        p2x1_plot.set_title('P2X1 Phase Space')
        p2x1_plot.set_xlabel('Position [$c/ \omega_{p}$]')
        p2x1_plot.set_ylabel('Proper Velocity $\gamma v_2$ [$c$]')
        plt.colorbar(p2x1_contour)

    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
    files=sorted(os.listdir(phase_space_dir))
    start=files[1].find('p1x1-electrons')+16
    end=files[1].find('.')
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(make_plot,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)
