import os
import glob
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
from ipywidgets import interact, interact_manual, Layout, fixed, FloatSlider, BoundedFloatText, IntSlider
import ipywidgets

import osiris
import osh5io
import analysis
import h5_utilities

interact_calc=interact_manual.options(manual_name="Make New Input and Run")    

title_font = {'fontname':'serif', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
axis_font = {'fontname':'serif', 'size':'20'}
plt.rc('font',size=16,family="serif")


def plot_theory(v0=10.0, density_ratio=1/100):
    alpha=np.linspace(0,20,num=200)
    rmass=density_ratio
    growth_rate=osiris.buneman_growth_rate(alpha,rmass)

    growth_rate_func=interp1d(alpha,np.abs(growth_rate),kind='cubic')

    #v0=10.0

    c=rmass


    karray=np.arange(0.0005,1,0.005)
    nk=49
    growth_rate=np.zeros(nk)
    growth_rate=growth_rate_func(karray*v0)
    plt.figure(figsize=(8,5))
    plt.plot(karray,growth_rate,label='Theory: $v_0 = '+repr(v0)+'$,\n'+'den ratio $n_b/n_0 = '+repr(c)+'$')

    plt.xlabel('Wavenumber [$1/\Delta x$]',**axis_font)
    plt.ylabel('Growth Rate [$\omega_{pe}$]',**axis_font)
    plt.legend()
    plt.show()
    
def run_upic(output_directory, inputfile):
    osiris.run_upic_es(rundir=output_directory,inputfile=inputfile)
    
def plot_t_vs_k(output_directory):

    workdir = os.getcwd()
    dirname = output_directory
    filename = workdir+'/'+dirname+'/Ex.h5'
    # print(filename)
    test4 = h5_utilities.read_hdf(filename)
    # here we fourier analyze the data in space
    #
    # k_data=np.fft.fft(test.data,axis=1)
    k_data=np.fft.fft(test4.data,axis=1)
    # k_data_2=np.fft.fft(k_data,axis=0)

    test4.data=np.abs(k_data)

    test4.axes[0].axis_max=2.0*3.1415926

    # test4.data=np.log10(np.real(test4.data)+1e-10)
    plt.figure(figsize=(8,5))
    analysis.plotme(test4)
    # k_bound=0.13
    k_max=0.1
    # plt.plot([k_bound,k_bound],[0,200],'b--',label='Instability Boundary')
    plt.plot([k_max,k_max],[0,200],'r--',label='Peak Location')
    plt.xlim(0,1)
    plt.ylim(0,190)
    plt.xlabel('Wavenumber [$1/\Delta x$]')
    plt.ylabel('Time [$1/\omega_p$]')
    # plt.ylim(0,50)
    # plt.ylim(tlim[0],tlim[1])
    plt.legend(loc='lower right',prop={'size': 12})
    plt.show()
    
def compare_sim_with_theory(output_directory, v0, mode, density_ratio):

    alpha=np.linspace(0,20,num=200)
    rmass=density_ratio
    growth_rate=osiris.buneman_growth_rate(alpha,rmass)

    growth_rate_func=interp1d(alpha,np.abs(growth_rate),kind='cubic')

    workdir = os.getcwd()
    dirname = output_directory
    filename = workdir+'/'+dirname+'/Ex.h5'
    # print(filename)
    test4 = h5_utilities.read_hdf(filename)
    # here we fourier analyze the data in space
    #
    # k_data=np.fft.fft(test.data,axis=1)
    k_data=np.fft.fft(test4.data,axis=1)
    # k_data_2=np.fft.fft(k_data,axis=0)

    test4.data=np.abs(k_data)

    test4.axes[0].axis_max=2.0*3.1415926

    nx=test4.data.shape[1]
    nt=test4.data.shape[0]
    # print(repr(nt))
    dk=2*3.1415926/nx
    # print('Delta k = '+repr(dk))


    # To compare with theory, just specify the mode you want to look at here
    #
    display_mode = mode
    bracket = False
    #
    #

    #v0=10.0

    alpha = v0 * dk * (display_mode)
    # growth_rate = 0.0
    # if (alpha<np.sqrt(2)): 
    growth_rate=growth_rate_func(alpha)[()]

    taxis=np.linspace(0,test4.axes[1].axis_max,nt)
    stream_theory=np.zeros(nt)
    stream_theory_plus=np.zeros(nt)
    stream_theory_minus=np.zeros(nt)
    init_amplitude=1e-7
    for it in range(0,nt):
        stream_theory[it]=init_amplitude*np.exp(growth_rate*taxis[it])
        stream_theory_plus[it]=init_amplitude*np.exp(1.15*growth_rate*taxis[it])
        stream_theory_minus[it]=init_amplitude*np.exp(0.85*growth_rate*taxis[it])

    plt.figure(figsize=(8,5))
    plt.semilogy(taxis,test4.data[:,display_mode],label='PIC simulation, mode ='+repr(display_mode))
    plt.semilogy(taxis,stream_theory,'r',label='theory, growth rate ='+'%.3f'%growth_rate)

    if (bracket):
        plt.semilogy(taxis,stream_theory_plus,'g.')

        plt.semilogy(taxis,stream_theory_minus,'g.')


    plt.ylim((1e-7,1000))
    plt.legend()
    plt.xlabel('Time $[1/\omega_{pe}]$',**axis_font)
    plt.ylabel('Mode amplitude [a.u.]', **axis_font)


    plt.show()
    
def plot_potential_xt(output_directory):
    osiris.plot_xt_arb(rundir=output_directory, field='pot',tlim=[0,250])
    
def phasespace_movie(output_directory):
    
    rundir = output_directory
    
    def something(rundir,file_no):

        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        efield_dir=working_dir+'/DIAG/Ex/'
        phase_space_dir=working_dir+'/DIAG/Vx_x/'
        ex_prefix='Ex-0_'
        phase_prefix='vx_x_'
        plt.figure(figsize=(10,5))

        filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        filename2=efield_dir+ex_prefix+repr(file_no).zfill(6)+'.h5'

        #print(filename1)
        #print(filename2)

        phase_space=np.abs(osh5io.read_h5(filename1))
        # print(repr(phase_space))
        ex=osh5io.read_h5(filename2)

        phase_plot=plt.subplot(121)
        #print(repr(phase_space.axes[0].min))
        #print(repr(phase_space.axes[1].min))
        title=phase_space.data_attrs['LONG_NAME']
        time=phase_space.run_attrs['TIME'][0]
        ext_stuff=[phase_space.axes[1].min,phase_space.axes[1].max,phase_space.axes[0].min,phase_space.axes[0].max]
        phase_contour=plt.contourf(abs(phase_space)+1e-11,
                                   levels=[0.1,1,2,3,5,10,100,1000,100000],
                                   extent=ext_stuff,
                                   cmap='Spectral',
                                   vmin=1e-1,
                                   vmax=100000,
                                   norm=matplotlib.colors.LogNorm(vmin=0.1,vmax=100000))
        phase_plot.set_title('Phase Space' +' , t='+repr(time)+' $\omega_{pe}^{-1}$')
        phase_plot.set_xlabel('Position [$\Delta x$]')
        phase_plot.set_ylabel('Velocity [$\omega_{pe} \Delta x$]')
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        plt.contour(phase_space,
                    levels=[0.1,1,2,3,5,10,100,1000,100000],
                    extent=ext_stuff,
                    colors='black',
                    linestyles='dashed')
        plt.colorbar(phase_contour)
        ex_plot = plt.subplot(122)

        plt.plot(ex[0,:])
        plt.ylim([-2,2])
        ex_plot.set_xlabel('Position [$\Delta x$]')
        ex_plot.set_ylabel('Electric Field')
        plt.tight_layout()
        plt.show()
#2345
    my_path=os.getcwd()
    working_dir=my_path+'/'+rundir
    phase_space_dir=working_dir+'/DIAG/Vx_x/'
    files=sorted(os.listdir(phase_space_dir))
    start=files[1].find('_x_')+3
    end=files[1].find('.')
    #print(files[1][start:end])
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(something,rundir=fixed(rundir),file_no=IntSlider(min=0,max=file_max,step=file_interval,value=0))
    

def wcb_deck_maker(iname='wcb.txt', oname='case1.txt', vdx=3.0, rden=0.01, vth0 = 1, vthb =0.01,
             tend=200):


    with open(iname) as osdata:
        data = osdata.readlines()
        
    BEAM_ELECTRONS = int(rden*262144)

    for i in range(len(data)):
        if 'VX0 =' in data[i]:
            data[i] = ' VDX = '+str(vdx)+',\n'
        if 'VTX' in data[i]:
            data[i] = ' VTX = '+str(vth0)+',\n'
            
        if 'VTDX' in data[i]:
            data[i] = ' VTDX = '+str(vthb)+'\n'
            
        if 'NPXB' in data[i]:
            data[i] = 'NPXB = '+str(BEAM_ELECTRONS)+'\n'
        if 'TEND' in data[i]:
            data[i] = ' TEND = '+str(tend)+',\n'

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running UPIC in directory '+dirname+'...')
    osiris.run_upic_es(rundir=dirname,inputfile=oname)
    outdirname=oname.split(".")[0]
    print(outdirname)
    # e_history=energy_history(dirname=outdirname)
    # taxis=np.arange(len(e_history))*0.2
    # plt.plot(taxis,e_history)
    # plt.title('Energy Deviation vs Time (in %)')
    # plt.xlabel('Time ($\omega_p^{-1}$)')   
    # plt.show()
 #
    phasespace_movie(output_directory=dirname)
    
    print('Done')

def wcb_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = ipywidgets.Text(value='wcb.txt', description='Template Input File:',style=style,layout=layout)
    b = ipywidgets.Text(value='case1.txt', description='New Output File:',style=style,layout=layout)
    c = ipywidgets.BoundedFloatText(value=3, min=0.0, max=10.0, description='Electron Drift Velocity:',style=style,layout=layout)
    d = ipywidgets.BoundedFloatText(value=0.01,min=0.01,max=1,step=0.01,description='Density Ratio (n_b/n_0):',style=style,layout=layout)
    e = ipywidgets.FloatText(value=1.0,description='VTH (of the plasma):',style=style,layout=layout)
    f = ipywidgets.BoundedFloatText(value=0.1,min=0,max=1,step=0.1,description='VTH (of the beam):',style=style,layout=layout)
    g = ipywidgets.FloatText(value=200.0,description='TEND or the Total Simulation Time:',style=style,layout=layout)
  

    im = interact_calc(wcb_deck_maker, iname=a, oname=b, vdx=c, rden=d, vth0 = e, vthb = f, tend=g);
    
    im.widget.manual_button.layout.width='300px'
