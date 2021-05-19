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


def run_upic(output_directory, inputfile):
    osiris.run_upic_es(rundir=output_directory, inputfile=inputfile)
    
    
def plot_w_vs_k(output_directory, v0):

    dirname = output_directory

    #v0=3
    nk=100
    k_array=np.linspace(0,2,num=nk)
    omega_plus=np.zeros(nk)
    omega_minus=np.zeros(nk)
    # plt.figure(figsize=(10,10))
    for ik in range(0,nk):
        omega_minus[ik]=-np.sqrt(2)+v0*k_array[ik]
        omega_plus[ik]=np.sqrt(2)+v0*k_array[ik]

    osiris.plot_wk_arb(rundir=dirname, field='pot',TITLE='pot',wlim=5, klim=1,plot_show=False)
    plt.plot(k_array,omega_plus,'b-',label='positive roots')
    plt.plot(k_array,omega_minus,'r.',label='negative roots')
    plt.legend()
    plt.show()

def plot_potential_xt(output_directory, tlim=[0,50], **kwargs):
    osiris.plot_xt_arb(rundir=output_directory, field='pot', tlim=tlim, **kwargs)
    
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
        if np.amax(abs(ex)) > 2:
            plt.ylim([-np.amax(abs(ex)),np.amax(abs(ex))])
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


def tstream_root_minus_r( k, v0, omegap):
    alpha=k*v0/omegap
    if (alpha > np.sqrt(2)):
        result = omegap*np.sqrt(1+alpha*alpha-np.sqrt(1+4*alpha*alpha))
    else:
        result = 0
    return result

def tstream_root_minus_i(k, v0, omegap):
    alpha=k*v0/omegap
    if (alpha < np.sqrt(2)):
        result = omegap*np.sqrt(np.sqrt(1+4*alpha*alpha)-1-alpha*alpha)
    else:
        result = 0
    return result

def tstream_plot_theory(v0,nx,kmin,kmax):

    # first let's define the k-axis

################################################################
################################################################
## need the simulation size to make the simulation curve
################################################################
################################################################

    nk=100
    karray=np.linspace(kmin,kmax,num=nk)
    k_pic_array=np.arange(kmin,kmax,2*3.1415926/nx)
    nmodes=k_pic_array.shape[0]
    omega_pic=np.zeros(nmodes)

    omega_plus=np.zeros(nk)
    omega_minus_r=np.zeros(nk)
    omega_minus_i=np.zeros(nk)

    # nk=karray.shape[0]
    #
    # using UPIC-ES normalization
    #
    omegap=1.0


    for i in range(0,nk):

        alpha=v0*karray[i]/omegap
        omega_plus[i]=omegap*np.sqrt(1+alpha*alpha+np.sqrt(1+4*alpha*alpha))
        omega_minus_r[i]=tstream_root_minus_r(karray[i],v0,omegap)
        omega_minus_i[i]=tstream_root_minus_i(karray[i],v0,omegap)

    for i in range(0,nmodes):
        omega_pic[i]=tstream_root_minus_i(k_pic_array[i],v0,omegap)

    plt.figure(figsize=(8,5))
    plt.plot(karray,omega_plus,'b-.',label = 'real root +')
    plt.plot(karray,omega_minus_i,'r',label = 'growth rate')
    plt.plot(karray,omega_minus_r,'b-.',label = 'real root - ')

    plt.plot(k_pic_array,omega_pic,'co',label='PIC Modes',markersize=15)
    plt.xlabel('wave number $[1/\Delta x]$')
    plt.ylabel('frequency $[\omega_{pe}]$')
    plt.title('Two Stream Theory in Simulation Units')
    plt.xlim((kmin,kmax))
    plt.ylim((0,3))
    plt.legend()

    plt.show()


def plot_t_vs_k(output_directory, v0=9.0, tlim=[0,80]):

    klim=5
    #tlim=80
    PATH = os.getcwd() + '/' + output_directory +'/'+ 'Ex.h5'
    hdf5_data = h5_utilities.read_hdf(PATH)

    k_data=np.fft.fft(hdf5_data.data,axis=1)
    hdf5_data.data=np.abs(k_data)

    hdf5_data.axes[0].axis_max=2.0*3.1415926*v0

    N=100
    dt = float(tlim[1])/N
    tvals=np.arange(0,tlim[1],dt)
    kvals=np.zeros(N)
    kpeak_vals=np.zeros(N)
    for i in range(0,N):
        kvals[i]=np.sqrt(2)
        kpeak_vals[i]=0.85

    plt.figure(figsize=(8,5))
    analysis.plotme(hdf5_data)
    plt.plot(kvals,tvals,'b--',label='Instability Boundary')
    plt.plot(kpeak_vals,tvals,'r--',label='Peak Location')

    plt.title('Ex t-k space' )

    plt.xlabel(' α ',**axis_font)
    plt.ylabel(' Time  [$1/ \omega_{pe}$]',**axis_font)
    plt.xlim(0,klim)
    plt.ylim(tlim)
    plt.legend()
    plt.show()


def compare_sim_with_theory(output_directory, modemin=1, modemax=5, v0=1, init_amplitude=1e-5, tlim=[0,80]):

    rundir = output_directory
    field = 'Ex'
    #tlim = 80
    
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = h5_utilities.read_hdf(PATH)

    k_data=np.fft.fft(hdf5_data.data,axis=1)
    hdf5_data.data=np.abs(k_data)

    nx=hdf5_data.data.shape[1]
    nt=hdf5_data.data.shape[0]
    taxis=np.linspace(0,hdf5_data.axes[1].axis_max,nt)
    deltak=2.0*3.1415926/nx
    hdf5_data.axes[0].axis_max=2.0*3.1415926*v0

#     nplots=modemax-modemin+1
    nplots=modemax-modemin+2

    N=100

    plt.figure(figsize=(8,3*nplots+1))

    plt.subplot(nplots,1,1)
    for imode in range(modemin,modemax+1):
        plt.semilogy(taxis,hdf5_data.data[:,imode],label='mode '+repr(imode))
    plt.ylabel('Mode Amplitudes')
    plt.xlabel('Time [$1/ \omega_{p}$]')
    plt.legend()
    plt.xlim(tlim)
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})

    for imode in range(modemin,modemax+1):
        plt.subplot(nplots,1,imode+1)
        stream_theory=np.zeros(nt)
        growth_rate=tstream_root_minus_i(deltak*imode,v0,1.0)
        for it in range(0,nt):
            stream_theory[it]=init_amplitude*np.exp(growth_rate*taxis[it])
        plt.semilogy(taxis,hdf5_data.data[:,imode],label='PIC simulation, mode ='+repr(imode))
        plt.semilogy(taxis,stream_theory,'r',label='theory, growth rate = {:.3f}'.format(growth_rate))
        plt.ylabel('mode'+repr(imode))
        plt.xlabel('Time [$1/ \omega_{p}$]')
        plt.legend()
        plt.xlim(tlim)

    plt.show()


def twostream_deck_maker(iname='twostream.txt', oname='case1.txt', n1=1, vx0=-3.0, vdx=3.0, n2n1=1.0, 
             tend=200, indx=8):

    with open(iname) as osdata:
        data = osdata.readlines()
        
    STREAM2_ELECTRONS = int(n2n1*262144)

    for i in range(len(data)):
        if 'VX0 =' in data[i]:
            data[i] = ' VX0 = '+str(vx0)+',\n'
        if 'VDX =' in data[i]:
            data[i] = ' VDX = '+str(vdx)+',\n'
        if 'NPXB' in data[i]:
            data[i] = 'NPXB = '+str(STREAM2_ELECTRONS)+'\n'
        if 'TEND' in data[i]:
            data[i] = ' TEND = '+str(tend)+',\n'
        if 'INDX' in data[i]:
            data[i] = ' INDX = '+str(indx)+',\n'

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
    
    print('Done')

def twostream_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = ipywidgets.Text(value='2stream.txt', description='Template Input File:',style=style,layout=layout)
    b = ipywidgets.Text(value='case1.txt', description='New Output File:',style=style,layout=layout)
    c = ipywidgets.BoundedFloatText(value=5, min=-30.0, max=30.0, description='Stream #1 Velocity (set first!):',style=style,layout=layout)
    d = ipywidgets.BoundedFloatText(value=-5, min=-30.0, max=30.0, description='Stream #2 Velocity:',style=style,layout=layout)
    e = ipywidgets.BoundedFloatText(value=1, min=0.0001, max=10.0, description='Density Ratio (n2/n1):',style=style,layout=layout)
    f = ipywidgets.BoundedFloatText(value=1, min=1, max=1, description='Stream #1 Density is fixed:',style=style,layout=layout)
    g = ipywidgets.FloatText(value=100.0,description='TEND or the Total Simulation Time:',style=style,layout=layout)
    h = ipywidgets.IntText(value=8,description='Box Length (In powers of 2!):',style=style,layout=layout)

    im = interact_calc(twostream_deck_maker, iname=a, oname=b, n1=f, vx0=c, vdx=d, n2n1=e, tend=g, indx=h);
    
    def c_handle_slider_change(change):
        d.value = -1/e.value*change.new
    c.observe(c_handle_slider_change, names='value')

    def d_handle_slider_change(change):
        e.value = -1/change.new*c.value
    d.observe(d_handle_slider_change, names='value')

    def e_handle_slider_change(change):
        d.value = -1/change.new*c.value
    e.observe(e_handle_slider_change, names='value')

    im.widget.manual_button.layout.width='300px'
