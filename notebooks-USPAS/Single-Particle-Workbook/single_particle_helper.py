import sys, glob, os, subprocess, re
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
import ipywidgets as widgets
interact_calc=interact_manual.options(manual_name="Make New Input and Run")
import osiris
from scipy import optimize
import numpy as np
import h5py
import matplotlib.pyplot as plt
    
def newifile(oname='single-part-1',field_solve='yee',dx1=0.2,dx2=20.0,
            dt_abs='Ratio (dt/t_courant)',dt=0.95,
            t_final=600.0,pusher='standard',uz0=0.0,a0=1.0,phi0=0.0,
            run_osiris=True,nproc=4,momentum_corr=True):
    
    if field_solve in ['fei','xu','yee-corr']:
        fname = 'single-part-fei.txt'
    else:
        fname = 'single-part-std.txt'

    # Remake the tags file based on the number of processors being used
    with open('tags-single-particle.txt') as osdata:
        data = osdata.readlines()
    data[1] = '{:d}, 1,'.format(nproc//2+1)
    with open('tags-single-particle.txt','w') as f:
        for line in data:
            f.write(line)

    with open(fname) as osdata:
        data = osdata.readlines()

    # Replace all parameters that don't depend on dt
    for i in range(len(data)):
        if 'node_number' in data[i]:
            data[i] = '  node_number(1:2) =  {:d}, 1,\n'.format(nproc)
        if 'nx_p' in data[i]:
            np_1 = np.around(200.0/dx1).astype(int)
            # Make even and divisible by nproc for correct particle loading
            divisor = np.lcm(2,nproc)
            np_1 = int( divisor * np.ceil( np_1 / divisor ) )
            dx1 = 200.0 / np_1
            data[i] = '  nx_p(1:2) =  {:d}, 12,\n'.format( np_1 )
        if 'xmin' in data[i]:
            data[i] = '  xmin(1:2) = -100.0, -{:f},\n'.format(6.0*dx2)
        if 'xmax' in data[i]:
            data[i] = '  xmax(1:2) =  100.0,  {:f},\n'.format(6.0*dx2)
        if ' x(1:5,1)' in data[i]:
            data[i] = '  x(1:5,1)  = -1.0, 0.0, {:f}, {:f}, {:f},\n'.format(dx1/2,dx1,dx1*2)
        if ' x(1:5,2)' in data[i]:
            data[i] = '  x(1:5,2)  = -1.0, 0.0, {:f}, {:f}, {:f},\n'.format(dx2/2,dx2,dx2*2)
        if 'tmax =' in data[i]:
            data[i] = '  tmax = '+str(t_final)+',\n'
        if 'push_type' in data[i]:
            data[i] = '  push_type = "'+pusher+'",\n'
        if ' a0 =' in data[i]:
            data[i] = '  a0 = '+str(a0)+',\n'
        if 'phase =' in data[i]:
            data[i] = '  phase = '+str(phi0)+',\n'
        if ' type' in data[i] and field_solve in ['xu','yee-corr']:
            data[i] = '  type = "'+field_solve+'",\n'

    if dt_abs == 'Ratio (dt/t_courant)':

        with open(oname+'.txt','w') as f:
            for line in data:
                f.write(line)

        # Calculate the courant limit given the above parameters
        # This is done by running OSIRIS with a large timestep and getting the max(dt) value
        if os.path.isfile('osiris-2D.e'):
            os_exec = './osiris-2D.e'
        else:
            os_exec = '/usr/local/osiris/osiris-2D.e'
        output = subprocess.run( [os_exec,"-t",oname+'.txt'], stderr=subprocess.DEVNULL, stdout=subprocess.PIPE )
        courant = float( re.search( r"max\(dt\).*([0-9]+\..*)\n", output.stdout.decode("utf-8") ).group(1) )

        dt_code = dt * courant

    else:

        dt_code = dt

    # Get proper initial conditions for the momentum (half time step back)
    # With this momentum, the true initial velocity in x1/x2 will average to 0
    if momentum_corr:
        ux0=0.0; uy0=0.0; t0=np.pi/2-phi0*np.pi/180.; z0=0.0;
        [u1_half,u2_half] = haines_initial(a0,ux0,uy0,uz0,t0,dt_code,z0)

    # Generate final input deck with new time step
    for i in range(len(data)):
        if 'dt ' in data[i]:
            data[i] = '  dt     =   '+str(dt_code)+',\n'
        if 'dtdx1' in data[i]:
            data[i] = '  dtdx1 = '+str(dt_code/dx1)+', ! dt/dx1\n'
        if 'ufl(1:3)' in data[i] and momentum_corr:
            data[i] = '  ufl(1:3) = '+str(u1_half)+', '+str( u2_half )+', 0.0,\n'
        if 'u10' in data[i]:
            data[i] = '  ! desired u10 = '+str(uz0)+',\n'
        if 'ndump ' in data[i]:
            total_dumps = 10 # Aim to dump data 10 times during the simulation
            at_least_dump = 100 # Dump at least every 100 time steps
            data[i] = ('  ndump  =   ' + 
                str( np.floor(t_final/dt_code/total_dumps).astype(int) )+',\n')
        if 'niter_tracks' in data[i]:
            total_points = 10000
            data[i] = ('  niter_tracks = ' + 
                str( np.ceil(t_final/dt_code/total_points).astype(int) )+',\n')

    with open(oname+'.txt','w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+'.txt is written.')
    if run_osiris:
        print('Running OSIRIS in directory '+oname+'...')
        osiris.runosiris_2d(rundir=oname,inputfile=oname+'.txt',print_out='yes',combine='no',np=nproc)

def single_particle_widget(run_osiris=True,nproc=4,momentum_corr=True):
    style = {'description_width': '350px'}
    layout = Layout(width='55%')

    a = widgets.Text(value='single-part-1', description='New output file:',style=style,layout=layout)
    b = widgets.Dropdown(options=['yee', 'fei', 'xu', 'yee-corr'],value='yee', description='Field solver:',style=style,layout=layout)
    c = widgets.BoundedFloatText(value=0.2, min=0.00001, max=3.0, description='dx1:',style=style,layout=layout)
    d = widgets.BoundedFloatText(value=20.0, min=0.00001, max=300.0, description='dx2:',style=style,layout=layout)
    e = widgets.Dropdown(options=['Ratio (dt/t_courant)', 'Absolute value'], value='Ratio (dt/t_courant)', description='dt specification',style=style,layout=layout)
    f = widgets.BoundedFloatText(value=0.999, min=0.0, max=0.999, description='dt:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=600.0, min=40.0, max=1e9, description='t_final:',style=style,layout=layout)
    h = widgets.Dropdown(options=['standard', 'vay', 'cond_vay', 'cary', 'fullrot', 'euler', 'petri'],value='standard',description='Pusher:',style=style,layout=layout)
    i = widgets.FloatText(value=0.0, description='uz0:',style=style,layout=layout)
    j = widgets.BoundedFloatText(value=1.0, min=0, max=1e9, description='a0:',style=style,layout=layout)
    k = widgets.BoundedFloatText(value=0.0, min=-180, max=180, description='phi0 (degrees):',style=style,layout=layout)

    def handle_dropdown_change(change):
        f.max = 0.999 if change.new == 'Ratio (dt/t_courant)' else 1e9
    e.observe(handle_dropdown_change, names='value')

    im = interact_calc(newifile, oname=a,field_solve=b,dx1=c,dx2=d,dt_abs=e,dt=f,t_final=g,pusher=h,uz0=i,a0=j,phi0=k,
        run_osiris=fixed(run_osiris),nproc=fixed(nproc),momentum_corr=fixed(momentum_corr));
    im.widget.manual_button.layout.width='250px'
    return a

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

    # Calculate the final s value that corresponds to the final t value
    # There can be error in this, so we calculate it in a while loop to make sure it's right
    tf_calc = 0.0
    count = 0
    max_iter = 10
    while not np.isclose(tf_calc,tf,rtol=1e-4,atol=1e-4) and count < max_iter:
        # Start guess at 0, then increase from there for large a0 values
        sf = optimize.root_scalar(t_haines,x0=tf*count/100,x1=tf).root

        s=np.linspace(0,sf,1000)
        x = a0/(g0*(1-bz0)) * ( np.sin( g0*(1-bz0)*s + phi0 ) - np.sin(phi0) ) - a0*s*np.cos(phi0) + g0*bx0*s
        z = 1./(2*g0*(1-bz0))*( 0.5*np.square(a0)*s + np.square(a0)/(4*g0*(1-bz0))*
                            ( np.sin(2*g0*(1-bz0)*s+2*phi0) - np.sin(2*phi0) ) + 
                            2*a0*(g0*bx0 - a0*np.cos(phi0))/(g0*(1-bz0))*( np.sin(g0*(1-bz0)*s+phi0) - np.sin(phi0) ) +
                            np.square(g0*bx0 - a0*np.cos(phi0))*s + s + np.square(g0*by0)*s ) - 0.5*g0*(1-bz0)*s
        t = z + g0*(1-bz0)*s
        tf_calc = t[-1]
        count += 1
        
    if count == max_iter:
        print("Could not calculate the correct t_final.  Aborting...")
        print("Desired t_final = ",tf,", calculated t_final = ",tf_calc)
        return

    px = a0*( np.cos(g0*(1-bz0)*s + phi0) - np.cos(phi0) ) + g0*bx0
    pz = 1./(2*g0*(1-bz0))*( np.square( -a0*(np.cos(g0*(1-bz0)*s + phi0) - np.cos(phi0)) - g0*bx0 ) + 
                            1 + np.square(g0*by0) ) - 0.5*g0*(1-bz0)
    g = np.sqrt(1+np.square(px)+np.square(pz))
    return [t,x,z,px,pz,g]

def haines_initial(a0,ux0,uy0,uz0,t0,dt,z0):
    g0 = np.sqrt( 1. + np.square(ux0) + np.square(uy0) + np.square(uz0) )
    bx0=ux0/g0; by0=uy0/g0; bz0=uz0/g0;
    phi0 = t0 - z0

    # Solve for the value of s for half time step back
    def t_haines(s):
        return (1./(2*g0*(1-bz0))*( 0.5*np.square(a0)*s + np.square(a0)/(4*g0*(1-bz0))*
                        ( np.sin(2*g0*(1-bz0)*s+2*phi0) - np.sin(2*phi0) ) + 
                        2*a0*(g0*bx0 - a0*np.cos(phi0))/(g0*(1-bz0))*( np.sin(g0*(1-bz0)*s+phi0) - np.sin(phi0) ) +
                        np.square(g0*bx0 - a0*np.cos(phi0))*s + s + np.square(g0*by0)*s ) - 0.5*g0*(1-bz0)*s + 
                        g0*(1-bz0)*s - (-dt/2) )

    # Calculate the initial s value that corresponds to -dt/2
    # There can be error in this, so we calculate it in a while loop to make sure it's right
    t = 0.0
    count = 0
    max_iter = 10
    while not np.isclose(t,-dt/2,rtol=1e-4,atol=1e-4) and count < max_iter:
        # Start second guess at 0, then decrease from there for large a0 values
        s = optimize.root_scalar(t_haines,x0=-dt/2,x1=-dt/2*count/100).root

        x = a0/(g0*(1-bz0)) * ( np.sin( g0*(1-bz0)*s + phi0 ) - np.sin(phi0) ) - a0*s*np.cos(phi0) + g0*bx0*s
        z = 1./(2*g0*(1-bz0))*( 0.5*np.square(a0)*s + np.square(a0)/(4*g0*(1-bz0))*
                            ( np.sin(2*g0*(1-bz0)*s+2*phi0) - np.sin(2*phi0) ) + 
                            2*a0*(g0*bx0 - a0*np.cos(phi0))/(g0*(1-bz0))*( np.sin(g0*(1-bz0)*s+phi0) - np.sin(phi0) ) +
                            np.square(g0*bx0 - a0*np.cos(phi0))*s + s + np.square(g0*by0)*s ) - 0.5*g0*(1-bz0)*s
        t = z + g0*(1-bz0)*s
        count += 1
        
    if count == max_iter:
        print("Could not calculate the correct t_initial.  Aborting...")
        print("Desired t_initial = ",-dt/2,", calculated t_initial = ",t)
        return

    # Get initial momentum a half time step back
    px = a0*( np.cos(g0*(1-bz0)*s + phi0) - np.cos(phi0) ) + g0*bx0
    pz = 1./(2*g0*(1-bz0))*( np.square( -a0*(np.cos(g0*(1-bz0)*s + phi0) - np.cos(phi0)) - g0*bx0 ) + 
                            1 + np.square(g0*by0) ) - 0.5*g0*(1-bz0)
    return [pz,px]

def grab_data(dirname):
    f=h5py.File(dirname+'/MS/TRACKS/electron-tracks.h5','r')
    t = f['data'][:,0]
    ene = f['data'][:,2]
    x1 = f['data'][:,3]
    x2 = f['data'][:,4]
    p1 = f['data'][:,5]
    p2 = f['data'][:,6]
    i_max = np.argmax(f['data'][:,1]==0) # Find where charge is 0, i.e., particle leaves
    f.close()

    x1 = x1-x1[0]
    x2 = x2-x2[0]

    with open(dirname+'.txt') as osdata:
        data = osdata.readlines()
    for i in range(len(data)):
        if 'xmax(1:2)' in data[i]:
            L_x2 = float(data[i].split(",")[-2])

    # Correct for periodicity jump in x2
    for i in np.arange(len(x2)-1):
        if x2[i+1]-x2[i]>L_x2:
            x2[i+1:] -= 2*L_x2
        elif x2[i+1]-x2[i]<-L_x2:
            x2[i+1:] += 2*L_x2

    return [t,x2,x1,p2,p1,ene,i_max]

def plot_data(dirnames,offset=None,theory=True,xlim_max=None,plot_z=False,save_fig=True):

    if type(dirnames) is not list:
        dirnames = [dirnames]

    # Get a0 and uz0 from first input deck
    dirname = dirnames[0]
    with open(dirname+'.txt') as osdata:
        data = osdata.readlines()
    for i in range(len(data)):
        if 'u10' in data[i]:
            uz0 = float(data[i].split(" ")[-1][:-2])
        if ' a0 =' in data[i]:
            a0 = float(data[i].split(" ")[-1][:-2])
        if 'phase = ' in data[i]:
            off = float(data[i].split(" ")[-1][:-2])*np.pi/180.

    if offset is not None:
        off = offset

    fig,axes=plt.subplots(1,5,figsize=(14,6),dpi=300)
    theory_ = False

    for dirname in dirnames:

        [t,x2,x1,p2,p1,ene,i_max] = grab_data(dirname)
        if xlim_max==None:
            xlim_max = np.max(t)
            l = len(t)
        else:
            if xlim_max >= np.max(t):
                l = len(t)
            else:
                l = np.argmax(t>xlim_max)

        # Don't plot values after the particle has left the box
        if i_max > 0:
            l = np.min([ l, i_max ])

        if dirname == dirnames[-1]:
            theory_ = theory
        if theory:
            ux0=0.0; uy0=0.0; t0=np.pi/2-off; z0=0.0;
            [tt,xx,zz,pxx,pzz,gg] = haines(a0,ux0,uy0,uz0,t0,xlim_max,z0)

        if plot_z:
            axes[0].plot(t[:l],x1[:l],label=dirname)
            if theory_: axes[0].plot(tt,zz,'--',label='theory')
            axes[0].set_ylabel('$z$ $[c/\omega_0]$')
        else:
            axes[0].plot(t[:l],x1[:l]-t[:l],label=dirname)
            if theory_: axes[0].plot(tt,zz-tt,'--',label='theory')
            axes[0].set_ylabel('$\\xi$ $[c/\omega_0]$')
        axes[0].set_xlabel('$t$ $[\omega_0^{-1}]$')
        axes[0].set_xlim([0,xlim_max])
        axes[0].legend(loc='upper right')

        axes[1].plot(t[:l],x2[:l])
        if theory_: axes[1].plot(tt,xx,'--')
        axes[1].set_xlabel('$t$ $[\omega_0^{-1}]$')
        axes[1].set_ylabel('$x$ $[c/\omega_0]$')
        axes[1].set_xlim([0,xlim_max])

        axes[2].plot(t[:l],p1[:l])
        if theory_: axes[2].plot(tt,pzz,'--')
        axes[2].set_xlabel('$t$ $[\omega_0^{-1}]$')
        axes[2].set_ylabel('$p_z$ $[m_ec]$')
        axes[2].set_xlim([0,xlim_max])

        axes[3].plot(t[:l],p2[:l])
        if theory_: axes[3].plot(tt,pxx,'--')
        axes[3].set_xlabel('$t$ $[\omega_0^{-1}]$')
        axes[3].set_ylabel('$p_x$ $[m_ec]$')
        axes[3].set_xlim([0,xlim_max])

        axes[4].plot(t[:l],ene[:l]+1)
        if theory_: axes[4].plot(tt,gg,'--')
        axes[4].set_xlabel('$t$ $[\omega_0^{-1}]$')
        axes[4].set_ylabel('$\gamma$')
        axes[4].set_xlim([0,xlim_max])

    plt.tight_layout()
    if save_fig:
        if len(dirnames) > 1:
            plt.savefig(dirnames[0]+'-comparison.png',dpi=300)
        else:
            plt.savefig(dirnames[0]+'/'+dirnames[0]+'.png',dpi=300)
    plt.show()