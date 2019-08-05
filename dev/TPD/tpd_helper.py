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

def newifile(iname='langdon-fixed.txt', oname='case1.txt',  
             a0=0.1, vth = 0.044, tmax=900.0 ):

    
    with open(iname) as osdata:
        data = osdata.readlines()
        


    for i in range(len(data)):
        if 'a0 =' in data[i] and 'omega0' not in data[i]:
            data[i] = 'a0 = '+str(a0)+',\n'
        if 'uth(1:3)' in data[i]:
            data[i] = 'uth(1:3) = '+str(vth)+' , '+str(vth)+' , '+str(vth)+',\n'
        if 'uth_bnd(1:3,1,1)' in data[i]:
            data[i] = 'uth_bnd(1:3,1,1) = '+str(vth)+' , '+str(vth)+' , '+str(vth)+',\n'
        if 'uth_bnd(1:3,2,1)' in data[i]:
            data[i] = 'uth_bnd(1:3,2,1) = '+str(vth)+' , '+str(vth)+' , '+str(vth)+',\n'
        if 'tmax =' in data[i]:
            data[i] = 'tmax ='+str(tmax)+'\n'

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running OSIRIS in directory '+dirname+'...')
    # osiris.runosiris(rundir=dirname,inputfile=oname,print_out='yes')
    osiris.runosiris_2d(rundir=dirname,inputfile=oname,print_out='yes',combine='no')
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

def tpd_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='langdon-fixed.txt', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value='case1.txt', description='New Output File:',style=style,layout=layout)
    c = widgets.FloatText(value=0.1,description='v_osc/c:',style=style,layout=layout)
    d = widgets.FloatText(value=0.044,description='vth/c:',style=style,layout=layout)
    e = widgets.FloatText(value=900.0,description='tmax:',style=style,layout=layout)
    
    # print('d='+repr(d))
    im = interact_calc(newifile, iname=a,oname=b,a0=c, vth=d, tmax=e);
    im.widget.manual_button.layout.width='250px'
    


    

    



def tpd_movie(rundir):
#2345
    import os


    def something(rundir,file_no):

        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        efield_dir=working_dir+'/MS/FLD/e1/'
        laser_dir = working_dir+'/MS/FLD/e2/'
        eden_dir = working_dir + '/MS/DENSITY/species_1/charge/'
        p1x1_dir=working_dir+'/MS/PHA/p1x1/species_1/'

        efield_prefix='e1-'
        laser_prefix='e2-'
        phase_prefix='p1x1-ions-'
        p1x1_prefix='p1x1-species_1-'
        eden_prefix='charge-species_1-'
        iden_prefix='charge-ions-'
        fig = plt.figure(figsize=(12,16) )

        filename2=eden_dir+eden_prefix+repr(file_no).zfill(6)+'.h5'
        filename3=efield_dir+efield_prefix+repr(file_no).zfill(6)+'.h5'
        filename4=laser_dir+laser_prefix+repr(file_no).zfill(6)+'.h5'
        filename5=p1x1_dir+p1x1_prefix+repr(file_no).zfill(6)+'.h5'

        #print(filename1)
        #print(filename2)

        # print(repr(phase_space))
        # eden=osh5io.read_h5(filename2)
        ex = osh5io.read_h5(filename3)
        ey = osh5io.read_h5(filename4)
        phase_space=np.abs(osh5io.read_h5(filename5))
   

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
                    levels=[0.00001*data_max,0.0001*data_max,0.001*data_max,0.01*data_max,0.05*data_max,0.1*data_max,0.2*data_max,0.5*data_max],
                    extent=ext_stuff,cmap='Spectral',vmin=1e-5*data_max,vmax=1.5*data_max,
                    norm=colors.LogNorm(vmin=0.00001*data_max,vmax=1.5*data_max))
        phase_plot.set_title('Electron P1X1 Phase Space')
        phase_plot.set_xlabel('Position [$c / \omega_{p}$]')
        phase_plot.set_ylabel('Proper Velocity $\gamma v_1$ [ c ]')
        
        
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        # plt.contour(np.abs(phase_space+0.000001),levels=[0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1],extent=ext_stuff,colors='black',linestyles='dashed')
        plt.colorbar(phase_contour)
        
        
        # den_plot = plt.subplot(223)
        # osh5vis.osplot(eden,title='Electron Density')
        
        
        # for i in range(ex.shape[0]-2,-1,-1):
        #     ex[i]=ex[i+1] + ex.axes[0].increment*ex[i]
        ex_plot = plt.subplot(222)
        # ext_stuff=[ex.axes[1].min,ex.axes[1].max,ex.axes[0].min,ex.axes[0].max]
        
        # ex_plot.set_title=('Plasmons vs space')
        
        osh5vis.osimshow(ex,title='Wake E-field ')
        
        ey_plot = plt.subplot(221)
        
        
        osh5vis.osimshow(ey,title='Laser Electric Field')
        
        
        
#2345
    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/MS/PHA/p1x1/species_1/'
    files=sorted(os.listdir(phase_space_dir))
    #print(files[1])
    start=files[1].find('p1x1-species_1')+16
    end=files[1].find('.')
    #print(files[1][start:end])
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(something,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0), continuous_update=False)
    #something(rundir=rundir,file_no=20)
    
    

    