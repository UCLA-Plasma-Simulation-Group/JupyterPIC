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
from matplotlib import rc
rc('text', usetex=True)

style = {'description_width': '350px'}
layout = Layout(width='55%')
a = widgets.Text(value='single-part-1.txt', description='New output file:',style=style,layout=layout)
sigz = FloatSlider(value = 3.0, min=0,max=10,step=0.05,continuous_update=False,description = r'$(\xi_f - \xi_i) \ [c/\omega_p]$',style = style,layout=layout)
a_0 = FloatSlider(value = 1.0, min=0,max=1.0,step=0.05,continuous_update=False,description = r'$a_0 \ [c/\omega_p]$',style = style,layout=layout)
lamb = FloatSlider(value = 1.0, min=0,max=10,step=0.05,continuous_update=False,description = r'$\Lambda$',style = style,layout=layout)
r_i = FloatSlider(value = 1.0, min=0,max=10,step=0.01562,continuous_update=False,description = r'$r_i \ [c/\omega_p]$',style = style,layout=layout)
tmax = FloatSlider(value = 20, min=10,max=200,step=10 ,continuous_update=False,description = r'$t_{max} \   [1/\omega_p]$',style = style,layout=layout)


def newifile(oname='single-part-1.txt',sigz = 3.0, a_0=1.0, lamb = 1.0,tmax=20,r_i = 1.0):
    

    fname = 'single-part-std.txt'
    with open(fname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        if 'tmax =' in data[i]:
            data[i] = '  tmax ='+str(tmax)+',\n'
        if 'n_b' in data[i]:
            data[i] = 'density = ' + str(2*np.float(lamb)/a_0**2) + ',\n'
        if 'a_0' in data[i]:
            data[i] = 'x(1:6,2)  = 0.0,' + str(a_0) + ',' + str(a_0 + 0.0001) + ',30, 40,50,\n'
        if 'x_i' in data[i]:
            data[i] = 'x(1:6,1)  = -30.001,' +str(-1 * sigz - 0.001) + ',' + str(-1 * sigz) + ',0.0001, 0.001, 1.0,\n'
        if('r_i' in data[i]):
            data[i] = 'x(1:6,2)  = 0.0,' + str(r_i) + ',' + str(r_i + 1.0/32) + ',' + str(r_i + 1.0/32 + 0.0001) + ',99, 100.0,\n'
    #     if 'ufl(1:3)' in data[i]:
    #         data[i] = '  ufl(1:3) = '+str(uz0)+', 0.0, 0.0,\n'
    #     if ' a0 =' in data[i]:
    #         data[i] = '  a0 = '+str(a0)+',\n'
    #     if 'phase =' in data[i]:
    #         data[i] = '  phase = '+str(phi0)+',\n'

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running OSIRIS in directory '+dirname+'...')
    os.system('chmod u+x ./osiris-2D.e')
    osiris.runosiris_2d(rundir=dirname,inputfile=oname,print_out='yes',combine='no',np=4)

def single_particle_widget():
    
    im = interact_calc(newifile, oname=a,sigz= sigz, a_0=a_0, lamb=lamb,tmax=tmax,r_i = r_i);
    im.widget.manual_button.layout.width='250px'



def grab_data(dirname):
    f=h5py.File(dirname+'/MS/TRACKS/electron-tracks.h5','r')

    t = f['data'][:,0]
    ene = f['data'][:,2]
    x1 = f['data'][:,3]
    x2 = f['data'][:,4]
    p1 = f['data'][:,5]
    p2 = f['data'][:,6]
    f.close()

    return [t,x2,x1,p2,p1,ene]

def plot_data(dirname,off=0.0,theory=True,xlim_max=None):
    rc('text', usetex=False)
    plt.rcParams.update({'font.size': 14})
    [t,x2,x1,p2,p1,ene] = grab_data(dirname)
    if xlim_max==None:
        tf = np.max(t)
    else:
        tf = xlim_max


    if xlim_max==None:
        xlim_max = tf
        l = len(t)
    else:
        if xlim_max >= np.max(t):
            l = len(t)
        else:
            l = np.argmax(t>xlim_max)

    xi = t[:l]-x1[:l]
    first_in = np.argmax(xi > 0)
    first_out = np.argmax(xi > sigz.value)
    if(first_out == 0):
        print('Increase the simulation time! The particle did not pass through the beam driver completely!')
    plt.figure(figsize=(14,14),dpi=200)
    
    ax = plt.subplot(321)
    plt.xlim(-2,12)
    plt.ylim(0,np.max(x2[:l]))
    plt.plot(xi,x2[:l],label='Simulation')

    plt.plot(sigz.value*np.ones(10),(np.arange(10))/9.0 * x2[first_out],'k--', label = r'$\xi_f$' )
    plt.plot(- np.arange(10) + sigz.value,(np.ones(10)) * x2[first_out],'k--' )
    plt.plot(sigz.value*np.zeros(10),(np.arange(10))/9.0 * x2[first_in],'m--', label = r'$\xi_i$' )
    # plt.plot(- np.arange(10) ,(np.ones(10)) * x2[first_in],'m--' )
    plt.legend()
    plt.text(0.05,x2[first_out] * 1.05, r'$\Delta r =$' + str(x2[first_out]-x2[first_in])[:3] )
        
    plt.ylabel(r'$r$ $[c/\omega_0]$')
    plt.xlabel(r'$\xi$ $[c/\omega_0]$')
    plt.legend()
    plt.minorticks_on()

    plt.subplot(322)
    plt.plot(p1[:l],0.5 * p2[:l]**2 ,label='Simulation')
    plt.plot(p1[:l],p1[:l], 'r--',label='Theory Eq. (21)')
    plt.legend()
    plt.xlabel(r'$p_z$ $[m_ec]$')
    plt.ylabel(r'$\frac{1}{2} p_r^2$ $[m_ec]$')
    plt.minorticks_on()

    plt.subplot(323)
    plt.plot(xi,ene[:l] +1 -p1[:l], label='Simulation')
    plt.ylim(0.9 *(ene[0] +1 -p1[0]), 1.1 * (ene[0] + 1 - p1[0]))
    plt.plot(xi,1 *np.ones(l), 'r--',label='Theory Eq. (17)')
    plt.legend()
    # plt.ylim(0.9,1.1)
    plt.xlabel(r'$\xi$ $[c/\omega_0]$')
    plt.ylabel(r'$\gamma - p_z$ $[m_ec]$')
    plt.minorticks_on()

    plt.subplot(324)
    plt.plot(xi,p2[:l],label='Simulation')
    ## Eq (36). solution
    if(first_out > 0):
        pr_solution = np.sqrt(2*lamb.value * np.log(x2[first_out]/x2[first_in]))
    else: 
        pr_solution = 0
    ## Student solution 
    ## needed variables
    ## lamb.value gives lambda
    ## a_0.value gives the driver radius
    ## Fill in these two lines
    use_student_solution = True
    if(first_out > 0):
        student_solution = np.sqrt(2*lamb.value * (np.log(x2[first_out]/a_0.value) + 0.5 *(1-x2[first_in]**2/a_0.value**2)))
    else:
        student_solution = 0
    ## 
    ## 



    if(first_out > 0):
        plt.plot(np.arange(int(xi[-1]+1)), np.ones(int(xi[-1]+1))*pr_solution,'r--',label=r'Theory Eq. (36)' )
        if(use_student_solution):
            plt.plot(np.arange(int(xi[-1]+1)), np.ones(int(xi[-1]+1))*student_solution,'b--',label=r'Student Solution' )
    else:
        plt.plot(np.arange(int(xi[-1]+1)), np.ones(int(xi[-1]+1))*pr_solution,'r--',label=r'Theory Eq. (36)' )
        if(use_student_solution):
            plt.plot(np.arange(int(xi[-1]+1)), np.ones(int(xi[-1]+1))*student_solution,'b--',label=r'Student Solution' )

    plt.xlabel(r'$\xi$ $[c/\omega_0]$')
    plt.legend()
    plt.ylabel(r'$p_r$')

    plt.subplot(325)
    plt.plot(t[:l],xi, label='Simulation')
    plt.legend()
    # plt.ylim(0.9,1.1)
    plt.ylabel(r'$\xi$ $[c/\omega_0]$')
    plt.xlabel(r'$t$ $[1/\omega_0]$')
    plt.minorticks_on()

    plt.subplot(326)
    plt.plot(t[:l],x2[:l], label='Simulation')
    plt.legend()
    # plt.ylim(0.9,1.1)
    plt.ylabel(r'$r$ $[c/\omega_0]$')
    plt.xlabel(r'$t$ $[1/\omega_0]$')
    plt.minorticks_on()

    plt.tight_layout()
    plt.minorticks_on()
    plt.show()
