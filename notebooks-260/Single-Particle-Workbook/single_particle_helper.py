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
    
def newifile(oname='single-part-1.txt',field_solve='yee',t_final=600.0,
            pusher='standard',uz0=0.0,a0=1.0,phi0=0.0):
    
    if field_solve == 'fei':
        fname = 'single-part-fei.txt'
    else:
        fname = 'single-part-std.txt'
    with open(fname) as osdata:
        data = osdata.readlines()

    for i in range(len(data)):
        if 'tmax =' in data[i]:
            data[i] = '  tmax ='+str(t_final)+',\n'
        if 'push_type' in data[i]:
            data[i] = '  push_type = "'+pusher+'",\n'
        if 'ufl(1:3)' in data[i]:
            data[i] = '  ufl(1:3) = '+str(uz0)+', 0.0, 0.0,\n'
        if ' a0 =' in data[i]:
            data[i] = '  a0 = '+str(a0)+',\n'
        if 'phase =' in data[i]:
            data[i] = '  phase = '+str(phi0)+',\n'

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = oname.strip('.txt')
    print('Running OSIRIS in directory '+dirname+'...')
    osiris.runosiris_2d(rundir=dirname,inputfile=oname,print_out='yes',combine='no',np=4)

def single_particle_widget():
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='single-part-1.txt', description='New output file:',style=style,layout=layout)
    b = widgets.Dropdown(options=['yee', 'fei'],value='yee', description='Field solver:',style=style,layout=layout)
    c = widgets.BoundedFloatText(value=600.0, min=0.0, max=1e9, description='t_final:',style=style,layout=layout)
    d = widgets.Dropdown(options=['standard', 'vay', 'cond_vay', 'cary'],value='standard',description='Pusher:',style=style,layout=layout)
    e = widgets.FloatText(value=0.0, description='uz0:',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=1.0, min=0, max=1e3, description='a0:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=0.0, min=-180, max=180, description='phi0 (degrees):',style=style,layout=layout)

    im = interact_calc(newifile, oname=a,field_solve=b,t_final=c,pusher=d,uz0=e,a0=f,phi0=g);
    im.widget.manual_button.layout.width='250px'

def haines(a0,ux0,uy0,uz0,t0,tf,z0):
    # Parameters
    # Ex = E0 sin(wt - kz)
    # a0 = laser amplitude
    # g0 = initially gamma of the particle
    # u[xyz]0 = normalized initial momenta (i.e., proper velocities, gamma*v)
    # t0 = initial time when the EM-wave hits the particle (can be thought of as phase of laser)
    # z0 = initial position of the particle
    g0 = np.sqrt( 1. + np.square(ux0) + np.square(uy0) + np.square(uz0) )
    bx0=ux0/g0; by0=uy0/g0; bz0=uz0/g0;

    phi0 = t0 - z0
    
    # Solve for the final value of s for the desired final value of time
    def t_haines(s):
        return (1./(2*g0*(1-bz0))*( 0.5*np.square(a0)*s + np.square(a0)/(4*g0*(1-bz0))*
                        ( np.sin(2*g0*(1-bz0)*s+2*phi0) - np.sin(2*phi0) ) + 
                        2*a0*(g0*bx0 - a0*np.cos(phi0))/(g0*(1-bz0))*( np.sin(g0*(1-bz0)*s+phi0) - np.sin(phi0) ) +
                        np.square(g0*bx0 - a0*np.cos(phi0))*s + s + np.square(g0*by0)*s ) - 0.5*g0*(1-bz0)*s + 
                        g0*(1-bz0)*s - tf)
    sf = optimize.root_scalar(t_haines,x0=0,x1=tf).root
    
    s=np.linspace(0,sf,1000)
    x = a0/(g0*(1-bz0)) * ( np.sin( g0*(1-bz0)*s + phi0 ) - np.sin(phi0) ) - a0*s*np.cos(phi0) + g0*bx0*s
    z = 1./(2*g0*(1-bz0))*( 0.5*np.square(a0)*s + np.square(a0)/(4*g0*(1-bz0))*
                        ( np.sin(2*g0*(1-bz0)*s+2*phi0) - np.sin(2*phi0) ) + 
                        2*a0*(g0*bx0 - a0*np.cos(phi0))/(g0*(1-bz0))*( np.sin(g0*(1-bz0)*s+phi0) - np.sin(phi0) ) +
                        np.square(g0*bx0 - a0*np.cos(phi0))*s + s + np.square(g0*by0)*s ) - 0.5*g0*(1-bz0)*s
    t = z + g0*(1-bz0)*s

    px = a0*( np.cos(g0*(1-bz0)*s + phi0) - np.cos(phi0) ) + g0*bx0
    pz = 1./(2*g0*(1-bz0))*( np.square( -a0*(np.cos(g0*(1-bz0)*s + phi0) - np.cos(phi0)) - g0*bx0 ) + 
                            1 + np.square(g0*by0) ) - 0.5*g0*(1-bz0)
    g = np.sqrt(1+np.square(px)+np.square(pz))
    return [t,x,z,px,pz,g]

def grab_data(dirname):
    f=h5py.File(dirname+'/MS/TRACKS/electron-tracks.h5','r')
    t = f['data'][:,0]
    ene = f['data'][:,2]
    x1 = f['data'][:,3]
    x2 = f['data'][:,4]
    p1 = f['data'][:,5]
    p2 = f['data'][:,6]
    f.close()

    x1 = x1-x1[0]
    x2 = x2-x2[0]

    # Correct for periodicity jump in x2
    for i in np.arange(len(x2)-1):
        if x2[i+1]-x2[i]>1.2:
            x2[i+1:] -= 2.4
        elif x2[i+1]-x2[i]<-1.2:
            x2[i+1:] += 2.4

    return [t,x2,x1,p2,p1,ene]

def plot_data(dirname,off=0.0,theory=True,xlim_max=None):
    # Get a0 and uz0 from input deck
    with open(dirname+'.txt') as osdata:
        data = osdata.readlines()
    for i in range(len(data)):
        if 'ufl(1:3)' in data[i]:
            uz0 = float(data[i].split(" ")[-3][:-1])
        if ' a0 =' in data[i]:
            a0 = float(data[i].split(" ")[-1][:-2])

    [t,x2,x1,p2,p1,ene] = grab_data(dirname)
    if xlim_max==None:
        tf = np.max(t)
    else:
        tf = xlim_max
    ux0=0.0; uy0=0.0; t0=np.pi/2+off; z0=0.0;
    [tt,xx,zz,pxx,pzz,gg] = haines(a0,ux0,uy0,uz0,t0,tf,z0)

    if xlim_max==None:
        xlim_max = tf
        l = len(t)
        ll = len(tt)
    else:
        if xlim_max >= np.max(t):
            l = len(t)
            ll = len(tt)
        else:
            l = np.argmax(t>xlim_max)
            ll = np.argmax(tt>xlim_max)

    plt.figure(figsize=(14,6),dpi=300)

    plt.subplot(151)
    plt.plot(t[:l],t[:l]-x1[:l],label='simulation')
    if theory: plt.plot(tt[:ll],tt[:ll]-zz[:ll],'--',label='theory')
    plt.xlabel('$t$ $[\omega_0^{-1}]$')
    plt.ylabel('$\\xi$ $[c/\omega_0]$')
    plt.xlim([0,xlim_max])
    plt.legend()

    plt.subplot(152)
    plt.plot(t[:l],x2[:l])
    if theory: plt.plot(tt[:ll],xx[:ll],'--')
    plt.xlabel('$t$ $[\omega_0^{-1}]$')
    plt.ylabel('$x$ $[c/\omega_0]$')
    plt.xlim([0,xlim_max])

    plt.subplot(153)
    plt.plot(t[:l],p1[:l])
    if theory: plt.plot(tt[:ll],pzz[:ll],'--')
    plt.xlabel('$t$ $[\omega_0^{-1}]$')
    plt.ylabel('$p_z$ $[m_ec]$')
    plt.xlim([0,xlim_max])

    plt.subplot(154)
    plt.plot(t[:l],p2[:l])
    if theory: plt.plot(tt[:ll],pxx[:ll],'--')
    plt.xlabel('$t$ $[\omega_0^{-1}]$')
    plt.ylabel('$p_x$ $[m_ec]$')
    plt.xlim([0,xlim_max])

    plt.subplot(155)
    plt.plot(t[:l],ene[:l]+1)
    if theory: plt.plot(tt[:ll],gg[:ll],'--')
    plt.xlabel('$t$ $[\omega_0^{-1}]$')
    plt.ylabel('$\gamma$')
    plt.xlim([0,xlim_max])

    plt.tight_layout()
    plt.show()