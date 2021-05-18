import os
import sys
import glob
import mpmath
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


def plot_chi(alpha, mass_ratio):
    x = np.arange(-2.0, 3.0, 0.001)
    chi = mass_ratio / x ** 2 + 1 / (x - alpha) ** 2
    
    plt.figure(figsize=(8,5))
    plt.plot(x, chi)
    plt.plot([-2.0, 3.0],[1.0, 1.0],'k--')
    plt.xlim([-2, 3])
    plt.ylim([0, 2])
    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('$\omega / \omega_{pe}$', size=20)
    plt.ylabel('$\chi(x)$', size = 20)
    plt.show()

    
def plot_chi_interactive():
    interact(plot_chi, alpha=FloatSlider(min=0.9, max=1.5, step=.01, description=r'$\alpha$', value = 1.12), mass_ratio=fixed(1.0/1836.0))
    


def buneman_growth_rate(alphaarray,rmass):

    nalpha=alphaarray.shape[0]

    alphamin=alphaarray[0]
    alphamax=alphaarray[nalpha-1]

    prev_root=complex(0,0)

    growth_rate=np.zeros(nalpha)
    growth_rate_r = np.zeros(nalpha)

    def buneman_disp(x):
        return (x**-(-rmass+x**2)*(x-alphaarray[0])**2)
    new_root=mpmath.findroot(buneman_disp,prev_root,solver='newton')
    growth_rate[0]=new_root.imag
    prev_root=complex(new_root.real,new_root.imag)
#    print(repr(prev_root))

    for i in range(1,nalpha):
        # print(repr(i))
        def buneman_disp2(x):
            return (x**2-(-rmass+x**2)*(x-alphaarray[i])**2)

        new_root =  mpmath.findroot(buneman_disp2, prev_root,solver='muller')
        growth_rate[i]=new_root.imag
        growth_rate_r[i] = new_root.real
        prev_root=complex(new_root.real,new_root.imag)

    return growth_rate, growth_rate_r

def plot_growth(mass_ratio):
    alpha = np.arange(0, 2.0, 0.01)
    # growth_rate = alpha
    # growth_rate = 1.0 * alpha*alpha
    growth_rate, growth_rate_r = buneman_growth_rate(alpha,mass_ratio)
    plt.figure(figsize=(8,5))
    plt.plot(alpha,growth_rate,'r.',label='Growth Rate')
    plt.plot(alpha,growth_rate_r,'g',label='Real Frequency')
#     plt.plot([-2.0, 3.0],[1.0, 1.0],'k--')
    plt.xlim([0, 1.2])
    plt.ylim([0, 0.25])
    plt.tick_params(axis='both', labelsize=18)
    plt.ylabel('$\omega / \omega_{pe}$', size=24)
    plt.xlabel(r'$\alpha$', size = 24)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

def plot_growth_interactive():    
#     interact(plot_growth, mass_ratio=FloatSlider(min=0.005, max=0.1, step=.002, description='$m/M$', value = 0.001))
    interact(plot_growth, mass_ratio=BoundedFloatText(min=0.001, max=0.1, step=0.001, description='$m/M$', value = 0.001))
    
    
def plot_theory(v0, mass_ratio):
    alpha=np.linspace(0,5,num=200)
    if v0 > 5.0:
        print('v0 should be <= 5.0')
        return
    #rmass=1.0/100.0
    rmass = mass_ratio
    #v0=3.0
    
    
    growth_rate=osiris.buneman_growth_rate(alpha,rmass)

    growth_rate_func=interp1d(alpha,np.abs(growth_rate),kind='cubic')

    karray=np.arange(0.01,1.0,0.01)
    nk=49
    growth_rate=np.zeros(nk)
    growth_rate=growth_rate_func(karray*v0)
    maxgr=max(growth_rate)
    print("Max growth rate is {:.3f}, occurring at k = {:.2f}".format(maxgr,karray[np.argmax(growth_rate)]))
    print("Instability edge is at k = {:.2f}".format(karray[np.where(growth_rate<0.001)[0]][0]))
    plt.figure(figsize=(8,5))
    plt.plot(karray,growth_rate,label='Theory: $v_0 = '+repr(v0)+'$,\n'+'mass ratio $m/M = '+repr(rmass)+'$')

    plt.xlabel('Wavenumber [$1/\Delta x$]',**axis_font)
    plt.ylabel('Growth Rate [$\omega_{pe}$]',**axis_font)
    plt.legend()
    plt.show()
    

def get_theory(v0, mass_ratio):
    alpha=np.linspace(0,5,num=200)
    if v0 > 5.0:
        print('v0 should be <= 5.0')
        return
    #rmass=1.0/100.0
    rmass = mass_ratio
    #v0=3.0
    
    
    growth_rate=osiris.buneman_growth_rate(alpha,rmass)

    growth_rate_func=interp1d(alpha,np.abs(growth_rate),kind='cubic')

    karray=np.arange(0.01,1.0,0.01)
    nk=49
    growth_rate=np.zeros(nk)
    growth_rate=growth_rate_func(karray*v0)
    maxgr=max(growth_rate)
    
    return maxgr, karray[np.argmax(growth_rate)], karray[np.where(growth_rate<0.001)[0]][0]

    
def run_upic(output_directory,inputfile):
    osiris.run_upic_es(rundir=output_directory,inputfile=inputfile)
    
    
def plot_t_vs_k(output_directory, v0=3.0, mass_ratio=1.0/100.0):

    workdir = os.getcwd()
    dirname = output_directory
    filename = workdir+'/'+dirname+'/Ex.h5'
    test4 = h5_utilities.read_hdf(filename)

    k_data=np.fft.fft(test4.data,axis=1)
    
    test4.data=np.abs(k_data)
    test4.axes[0].axis_max=2.0*3.1415926

    plt.figure(figsize=(8,6))
    analysis.plotme(test4)
    
    maxgr, kofmax, kedge = get_theory(v0, mass_ratio)
    
    #k_bound=0.446664
    #k_max=0.34713
    k_bound = kedge
    k_max = kofmax
    
    plt.plot([k_bound,k_bound],[0,200],'b--',label='Instability Boundary') 
    plt.plot([k_max,k_max],[0,200],'r--',label='Peak Location')
    plt.xlim(0,1)
    plt.ylim(0,190)
    plt.xlabel('Wavenumber [$1/\Delta x$]')
    plt.ylabel('Time [$1/\omega_p$]')
    # plt.ylim(0,50)
    # plt.ylim(tlim[0],tlim[1])
    plt.legend(loc='lower right',prop={'size': 12})
    plt.show()
    
def compare_sim_with_theory(output_directory, v0, mode, mass_ratio):
    
    alpha = np.arange(0, 2.0, 0.01)
    rmass=mass_ratio

    growth_rate=osiris.buneman_growth_rate(alpha,rmass)

    growth_rate_func=interp1d(alpha,np.abs(growth_rate),kind='cubic')

    workdir = os.getcwd()
    dirname = output_directory
    filename = workdir+'/'+dirname+'/Ex.h5'
    test4 = h5_utilities.read_hdf(filename)

    k_data=np.fft.fft(test4.data,axis=1)
    
    test4.data=np.abs(k_data)
    test4.axes[0].axis_max=2.0*3.1415926

    nx=test4.data.shape[1]
    nt=test4.data.shape[0]

    dk=2*3.1415926/nx

    display_mode = mode
    bracket = False

    #v0=3.0

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

    plt.figure(figsize=(8,6))
    plt.semilogy(taxis,test4.data[:,display_mode],label='PIC simulation, mode ='+repr(display_mode))
    plt.semilogy(taxis,stream_theory,'r',label='theory, growth rate ='+'%.3f'%growth_rate)

    if (bracket):
        plt.semilogy(taxis,stream_theory_plus,'g.')

        plt.semilogy(taxis,stream_theory_minus,'g.')

    plt.ylim((1e-7,1000))
    plt.legend()
    plt.xlabel('Time $[1/\omega_{pe}]$',**axis_font)
    plt.ylabel('Time History [a.u.]', **axis_font)

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
    

def buneman_deck_maker(iname='buneman.txt', oname='case1.txt', vx0=3.0, rmass=100,
             tend=200):

    with open(iname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        if 'VX0 =' in data[i]:
            data[i] = ' VX0 = '+str(vx0)+',\n'
        if 'WAVEW' in data[i]:
            data[i] = ' RMASS = '+str(rmass)+',\n'
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

    phasespace_movie(output_directory=dirname)
    buneman_plot_t_vs_k(output_directory=dirname, v0=vx0, mass_ratio=1/rmass)
    
    print('Done')

def buneman_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = ipywidgets.Text(value='buneman.txt', description='Template Input File:',style=style,layout=layout)
    b = ipywidgets.Text(value='case1.txt', description='New Output File:',style=style,layout=layout)
    c = ipywidgets.BoundedFloatText(value=3, min=0.0, max=10.0, description='Electron Drift Velocity:',style=style,layout=layout)
    d = ipywidgets.FloatText(value=100.0,description='Mass Ratio (M/m):',style=style,layout=layout)
    e = ipywidgets.FloatText(value=200.0,description='TEND or the Total Simulation Time:',style=style,layout=layout)

    im = interact_calc(buneman_deck_maker, iname=a,oname=b,VX0=c,RMASS=d, tend=e);
    
    im.widget.manual_button.layout.width='250px'
