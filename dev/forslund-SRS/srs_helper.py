import sys
b= sys.path
sys.path=['/home/jovyan/analysis'] + b
import osiris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from osh5vis import new_fig, osplot
from osh5io import *
import osh5io
import osh5vis
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

    
def newifile(iname='forslund-srs.txt', oname='case1.txt',  
             a0=0.1, uth=0.03,  n0 = 0.1, tmax=1800.0 ):


    
    with open(iname) as osdata:
        data = osdata.readlines()
        


    for i in range(len(data)):
        if 'uth(1:3) =' in data[i]:
            data[i] = 'uth(1:3) = '+str(uth)+' , '+str(uth)+' , '+str(uth)+',\n'
        if 'uth_bnd(1:3,1,1) =' in data[i]:
            data[i] = 'uth_bnd(1:3,1,1) = '+str(uth)+' , '+str(uth)+' , '+str(uth)+',\n'
        if 'uth_bnd(1:3,2,1) =' in data[i]:
            data[i] = 'uth_bnd(1:3,2,1) = '+str(uth)+' , '+str(uth)+' , '+str(uth)+',\n'
        if 'a0 =' in data[i] and 'omega0' not in data[i]:
            data[i] = 'a0 = '+str(a0)+',\n'
        if 'fx(1:6,1)' in data[i]:
            data[i] = 'fx(1:6,1) =  0.0, 0.0,' +str(n0) +' , '+str(n0)+' , 0.0, 0.0,'
        if 'tmax =' in data[i]:
            data[i] = 'tmax ='+str(tmax)+'\n'

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running OSIRIS in directory '+dirname+'...')
    osiris.runosiris(rundir=dirname,inputfile=oname,print_out='yes')
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

def srs_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='forslund-sbs.txt', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value='case1.txt', description='New Output File:',style=style,layout=layout)
    c = widgets.FloatText(value=0.1,description='a0:',style=style,layout=layout)
    d = widgets.FloatText(value=0.035,description='uth/c:',style=style,layout=layout)
    e = widgets.FloatText(value=0.1,description='n/n_c:',style=style,layout=layout)
    f = widgets.FloatText(value=1800.0,description='tmax:',style=style,layout=layout)

    
    im = interact_calc(newifile, iname=a,oname=b,a0=c,uth=d,n0 = e, tmax=f); 
    im.widget.manual_button.layout.width='250px'
    

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
    


    
def xt_and_energy_plot2(rundir, field='e2'):
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
    
    inputfile = rundir + '.txt'
    omega0 = find_omega_0(inputfile)
    print(omega0)
    vg = (1 - omega0**(-2))**(0.5) #normalized
    print(vg)
    t_dephase = pi / (2 * (1-vg)) 
    if field == 'e2':
        tplot = np.linspace(0, tlim[1], 100)
        xt = 24 - (1 - vg) * tplot
        plt.plot(xt,tplot,'k-')
        xeven = np.linspace(xlim[0], xlim[1], 100)
        t_dephase_plot = np.ones(100) * t_dephase
        plt.plot(xeven, t_dephase_plot,'k-')
        
    wake_back = np.ones(100) * (xlim[1] - (0.5 + pi))
    
    if field == 'e1':
        tplot = np.linspace(0, tlim[1], 100)
        xt = 23.5 - (1 - vg) * tplot
        plt.plot(xt,tplot,'k-')
        xeven = np.linspace(xlim[0], xlim[1], 100)
        t_dephase_plot = np.ones(100) * t_dephase
        plt.plot(xeven, t_dephase_plot,'k-')
        t_wake_plot = np.linspace(tlim[0], tlim[1], 100)
        plt.plot(wake_back, t_wake_plot,'k-')
    fig.show() 
    

def find_omega_0(iname='case0.txt'):

    with open(iname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        if 'omega0 =' in data[i]:
    #            pdb.set_trace()
            omega0 = data[i].replace('omega0 = ','')
            omega0 = omega0.replace(',\n', '')
            return float(omega0)
            ##something like return int(data[#] or where # is the indices where the number actually is
    #') 


def srs_movie(rundir):
#2345
    import os


    def something(rundir,file_no):

        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        efield_dir=working_dir+'/MS/FLD/e1/'
        laser_dir = working_dir+'/MS/FLD/e2/'
        eden_dir = working_dir + '/MS/DENSITY/electrons/charge/'
        iden_dir = working_dir + '/MS/DENSITY/ions/charge/'
        phase_space_dir=working_dir+'/MS/PHA/p1x1/ions/'
        p1x1_dir=working_dir+'/MS/PHA/p1x1/electrons/'

        efield_prefix='e1-'
        laser_prefix='e2-'
        phase_prefix='p1x1-ions-'
        p1x1_prefix='p1x1-electrons-'
        eden_prefix='charge-electrons-'
        iden_prefix='charge-ions-'
        fig = plt.figure(figsize=(12,16) )

        # filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        filename2=eden_dir+eden_prefix+repr(file_no).zfill(6)+'.h5'
        filename3=efield_dir+efield_prefix+repr(file_no).zfill(6)+'.h5'
        filename4=laser_dir+laser_prefix+repr(file_no).zfill(6)+'.h5'
        filename5=p1x1_dir+p1x1_prefix+repr(file_no).zfill(6)+'.h5'
        # filename6=iden_dir+iden_prefix+repr(file_no).zfill(6)+'.h5'

        #print(filename1)
        #print(filename2)

        phase_space=np.abs(osh5io.read_h5(filename5))
        # print(repr(phase_space))
        eden=osh5io.read_h5(filename2)
        ex = osh5io.read_h5(filename3)
        ey = osh5io.read_h5(filename4)
        # p1x1=np.abs(osh5io.read_h5(filename5))
        # iden = osh5io.read_h5(filename6)

        phase_plot=plt.subplot(224 )
        #print(repr(phase_space.axes[0].min))
        #print(repr(phase_space.axes[1].min))
        title=phase_space.data_attrs['LONG_NAME']
        time=phase_space.run_attrs['TIME'][0]

        fig.suptitle('Time = '+repr(time)+'$\omega_p^{-1}$',fontsize=24)
        ext_stuff=[phase_space.axes[1].min,phase_space.axes[1].max,phase_space.axes[0].min,phase_space.axes[0].max]
        data_max=max(np.abs(np.amax(phase_space)),100)
        #print(repr(data_max))
        phase_contour=plt.contourf(np.abs(phase_space+0.000000001),
                    levels=[0.000001*data_max,0.00001*data_max,0.0001*data_max,0.001*data_max,0.005*data_max,0.01*data_max,0.02*data_max,0.05*data_max],
                    extent=ext_stuff,cmap='Spectral',vmin=1e-6*data_max,vmax=1.5*data_max,
                    norm=colors.LogNorm(vmin=0.000001*data_max,vmax=1.5*data_max))
        phase_plot.set_title('Ion P1X1 Phase Space')
        phase_plot.set_xlabel('Position [$c / \omega_{p}$]')
        phase_plot.set_ylabel('Proper Velocity $\gamma v_1$ [ c ]')
        # second_x = plt.twinx()
        # second_x.plot(ex.axes[0],ex,'g',linestyle='-.')
        
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        # plt.contour(np.abs(phase_space+0.000001),levels=[0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1],extent=ext_stuff,colors='black',linestyles='dashed')
        plt.colorbar(phase_contour)
        
        
        den_plot = plt.subplot(223)
        
        osh5vis.osplot(eden,title='Electron Density')
       
        ex_plot = plt.subplot(222)
        
        osh5vis.osplot(ex,title='Wake E-field ',ylabel='$E_1 [m_e c^2/e]$')
        
        ey_plot = plt.subplot(221)
        
        
        osh5vis.osplot(ey,title='Laser Electric Field')
        
        
        
#2345
    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
    files=sorted(os.listdir(phase_space_dir))
    #print(files[1])
    start=files[1].find('p1x1-electrons')+16
    end=files[1].find('.')
    #print(files[1][start:end])
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(something,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)
    #something(rundir=rundir,file_no=20)
    

def scs_movie(rundir):
#2345
    import os


    def something(rundir,file_no):

        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        efield_dir=working_dir+'/MS/FLD/e1/'
        laser_dir = working_dir+'/MS/FLD/e2/'
        eden_dir = working_dir + '/MS/DENSITY/electrons/charge/'
        iden_dir = working_dir + '/MS/DENSITY/ions/charge/'
        phase_space_dir=working_dir+'/MS/PHA/p1x1/ions/'
        p1x1_dir=working_dir+'/MS/PHA/p1x1/electrons/'

        efield_prefix='e1-'
        laser_prefix='e2-'
        phase_prefix='p1x1-ions-'
        p1x1_prefix='p1x1-electrons-'
        eden_prefix='charge-electrons-'
        iden_prefix='charge-ions-'
        fig = plt.figure(figsize=(12,16) )

        # filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        filename2=eden_dir+eden_prefix+repr(file_no).zfill(6)+'.h5'
        filename3=efield_dir+efield_prefix+repr(file_no).zfill(6)+'.h5'
        filename4=laser_dir+laser_prefix+repr(file_no).zfill(6)+'.h5'
        filename5=p1x1_dir+p1x1_prefix+repr(file_no).zfill(6)+'.h5'
        # filename6=iden_dir+iden_prefix+repr(file_no).zfill(6)+'.h5'

        #print(filename1)
        #print(filename2)

        phase_space=np.abs(osh5io.read_h5(filename5))
        # print(repr(phase_space))
        eden=osh5io.read_h5(filename2)
        ex = osh5io.read_h5(filename3)
        ey = osh5io.read_h5(filename4)
        # p1x1=np.abs(osh5io.read_h5(filename5))
        # iden = osh5io.read_h5(filename6)

        phase_plot=plt.subplot(224 )
        #print(repr(phase_space.axes[0].min))
        #print(repr(phase_space.axes[1].min))
        title=phase_space.data_attrs['LONG_NAME']
        time=phase_space.run_attrs['TIME'][0]

        fig.suptitle('Time = '+repr(time)+'$\omega_p^{-1}$',fontsize=24)
        ext_stuff=[phase_space.axes[1].min,phase_space.axes[1].max,phase_space.axes[0].min,phase_space.axes[0].max]
        data_max=max(np.abs(np.amax(phase_space)),100)
        #print(repr(data_max))
        phase_contour=plt.contourf(np.abs(phase_space+0.000000001),
                    levels=[0.000001*data_max,0.00001*data_max,0.0001*data_max,0.001*data_max,0.005*data_max,0.01*data_max,0.02*data_max,0.05*data_max],
                    extent=ext_stuff,cmap='Spectral',vmin=1e-6*data_max,vmax=1.5*data_max,
                    norm=colors.LogNorm(vmin=0.000001*data_max,vmax=1.5*data_max))
        phase_plot.set_title('Ion P1X1 Phase Space')
        phase_plot.set_xlabel('Position [$c / \omega_{p}$]')
        phase_plot.set_ylabel('Proper Velocity $\gamma v_1$ [ c ]')
        # second_x = plt.twinx()
        # second_x.plot(ex.axes[0],ex,'g',linestyle='-.')
        
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        # plt.contour(np.abs(phase_space+0.000001),levels=[0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1],extent=ext_stuff,colors='black',linestyles='dashed')
        plt.colorbar(phase_contour)
        
        
        den_plot = plt.subplot(223)
        osh5vis.osplot(np.log(np.sum(np.abs(phase_space),axis=1)+0.001),title='f(v)')
        
       
        ex_plot = plt.subplot(222)
        
        osh5vis.osplot(ex,title='Wake E-field ',ylabel='$E_1 [m_e c^2/e]$')
        
        ey_plot = plt.subplot(221)
        
        
        osh5vis.osplot(ey,title='Laser Electric Field')
        
        
        
#2345
    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
    files=sorted(os.listdir(phase_space_dir))
    #print(files[1])
    start=files[1].find('p1x1-electrons')+16
    end=files[1].find('.')
    #print(files[1][start:end])
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(something,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)
    #something(rundir=rundir,file_no=20)
    
    

    