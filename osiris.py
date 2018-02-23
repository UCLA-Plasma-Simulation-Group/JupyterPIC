import os
import shutil
import subprocess
import IPython.display
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from h5_utilities import *
from analysis import *
from scipy.optimize import fsolve

def execute(cmd):

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
        
        
def run_upic_es(rundir='',inputfile='pinput2'):
    
    def combine_h5_2d(path, ex):
        in_file = workdir + '/' + path + '/' + ex + '/'
        out_file = workdir + '/' + ex + '.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_2d.py", in_file, out_file]):
            IPython.display.clear_output(wait=True)
#            print(path, end='')
            
    def combine_h5_iaw_2d():
        in_file = workdir + '/DIAG/IDen/'
        out_file = workdir + '/ions.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_2d.py", in_file, out_file]):
            IPython.display.clear_output(wait=True)
#            print(path, end='')
    
    workdir = os.getcwd()
    workdir += '/' + rundir
    print(workdir)

    if(not os.path.isdir(workdir)):
       os.mkdir(workdir)
    if(rundir != ''):
        shutil.copyfile(inputfile,workdir+'/pinput2')
    
    os.chdir(workdir)
        
    # run the upic-es executable    
    waittick = 0
    for path in execute(["/usr/local/beps/upic-es.out"]):
        waittick += 1
        if(waittick == 100):
            IPython.display.clear_output(wait=True)
            waittick = 0
            print(path, end='')
            
    
            
    # run the combine script
    print('combining Ex files')
    combine_h5_2d('DIAG', 'Ex')
    print('combining Ey files')
    combine_h5_2d('DIAG', 'Ey')
    print('combining pot files')
    combine_h5_2d('DIAG', 'pot')

    print('combine_h5_2d completed normally')
    
    # run combine on iaw data if present
    if (os.path.isdir(workdir + '/DIAG/IDen/')):
        print('combining IAW files')
        combine_h5_iaw_2d()
        print('combine_h5_iaw completed normally')
        
#     IPython.display.clear_output(wait=True)
    print('runbeps completed normally')
    os.chdir('../')
    
    return
    

def runosiris(rundir='',inputfile='osiris-input.txt'):

    def combine_h5_1d(ex):
        in_file = workdir + '/MS/FLD/' + ex + '/'
        out_file = workdir + '/' + ex + '.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_1d.py", in_file, out_file]):
            IPython.display.clear_output(wait=True)
#            print(path, end='')

    def combine_h5_iaw_1d():
        in_file = workdir + '/MS/DENSITY/ions/charge/'
        out_file = workdir + '/ions.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_1d.py", in_file, out_file]):
            IPython.display.clear_output(wait=True)
#            print(path, end='')

    workdir = os.getcwd()
    workdir += '/' + rundir
    print(workdir)

    # run osiris-1D.e executable
    if(not os.path.isdir(workdir)):
       os.mkdir(workdir)
    if(rundir != ''):
#        shutil.copyfile('osiris-1D.e',workdir+'/osiris-1D.e')
        shutil.copyfile(inputfile,workdir+'/osiris-input.txt')
    waittick = 0
    for path in execute(["osiris-1D.e","-w",workdir,"osiris-input.txt"]):
        waittick += 1
        if(waittick == 100):
            IPython.display.clear_output(wait=True)
            waittick = 0
            print(path, end='')

    # run combine_h5_util_1d.py script for e1/, e2/, e3/ (and iaw if applicable)
    print('combining E1 files')
    combine_h5_1d('e1')
    print('combining E2 files')
    combine_h5_1d('e2')
    print('combining E3 files')
    combine_h5_1d('e3')

    # run combine on iaw data if present
    if (os.path.isdir(workdir+'/MS/DENSITY/ions/charge')):
        print('combining IAW files')
        combine_h5_iaw_1d()
        
    IPython.display.clear_output(wait=True)
    print('runosiris completed normally')

    return


def field(rundir='',dataset='e1',time=0,space=-1,
    xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[], **kwargs):

    if(space != -1):
        plot_or = 1
        PATH = gen_path(rundir, plot_or)
        hdf5_data = read_hdf(PATH)
        plt.figure(figsize=(8,5))
        #plotme(hdf5_data.data[:,space], **kwargs)
        plotme(hdf5_data,hdf5_data.data[space,:])
        plt.title('temporal evolution of e' + str(plot_or) + ' at cell ' + str(space))
        plt.xlabel('t')
        plt.show()
        return


    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'FLD', dataset)
    files = sorted(os.listdir(odir))

    i = 0
    for j in range(len(files)):
        fhere = h5py.File(os.path.join(odir,files[j]), 'r')
        if(fhere.attrs['TIME'] >= time):
            i = j
            break

    fhere = h5py.File(os.path.join(odir,files[i]), 'r')

    plt.figure(figsize=(6, 3.2))
    plt.title(dataset+' at t = '+str(fhere.attrs['TIME']))
    plt.xlabel('$x_1 [c/\omega_p]$')
    plt.ylabel(dataset)

    xaxismin = fhere['AXIS']['AXIS1'][0]
    xaxismax = fhere['AXIS']['AXIS1'][1]

    nx = len(fhere[dataset][:])
    dx = (xaxismax-xaxismin)/nx

    plt.plot(np.arange(0,xaxismax,dx),fhere[dataset][:])

    if(xlim != [-1,-1]):
        plt.xlim(xlim)
    if(ylim != [-1,-1]):
        plt.ylim(ylim)
    if(zlim != [-1,-1]):
        plt.clim(zlim)

    plt.show()


def phasespace(rundir='',dataset='p1x1',species='electrons',time=0,
    xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[]):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'PHA', dataset, species)
    files = sorted(os.listdir(odir))

    i = 0
    for j in range(len(files)):
        fhere = h5py.File(os.path.join(odir,files[j]), 'r')
        if(fhere.attrs['TIME'] >= time):
            i = j
            break

    fhere = h5py.File(os.path.join(odir,files[i]), 'r')

    plt.figure(figsize=(6, 3.2))
    plt.title(dataset+' phasespace at t = '+str(fhere.attrs['TIME']))
    plt.xlabel('$x_1 [c/\omega_p]$')
    if(len(fhere['AXIS']) == 1):
        plt.ylabel('$n [n_0]$')
    if(len(fhere['AXIS']) == 2):
        plt.ylabel('$p_1 [m_ec]$')

    if(len(fhere['AXIS']) == 1):

        xaxismin = fhere['AXIS']['AXIS1'][0]
        xaxismax = fhere['AXIS']['AXIS1'][1]

        nx = len(fhere[dataset][:])
        dx = (xaxismax-xaxismin)/nx

        plt.plot(np.arange(0,xaxismax,dx),np.abs(fhere[dataset][:]))

    elif(len(fhere['AXIS']) == 2):

        xaxismin = fhere['AXIS']['AXIS1'][0]
        xaxismax = fhere['AXIS']['AXIS1'][1]
        yaxismin = fhere['AXIS']['AXIS2'][0]
        yaxismax = fhere['AXIS']['AXIS2'][1]

        plt.imshow(np.log(np.abs(fhere[dataset][:,:]+1e-12)),
                   aspect='auto',
                   extent=[xaxismin, xaxismax, yaxismin, yaxismax])
        plt.colorbar(orientation='vertical')


    if(xlim != [-1,-1]):
        plt.xlim(xlim)
    if(ylim != [-1,-1]):
        plt.ylim(ylim)
    if(zlim != [-1,-1]):
        plt.clim(zlim)

    plt.show()


def lineinteract(rundir='',dataset='e1',
    xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[]):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'FLD', dataset)
    files = sorted(os.listdir(odir))

    f0 = h5py.File(os.path.join(odir,files[0]), 'r')
    xaxismin = f0['AXIS']['AXIS1'][0]
    xaxismax = f0['AXIS']['AXIS1'][1]
    nx = len(f0[dataset][:])
    dx = (xaxismax-xaxismin)/nx
    xaxis = np.arange(0,xaxismax,dx)

    data = []
    for i in range(len(files)):
        fhere = h5py.File(os.path.join(odir,files[i]), 'r')
        data.append([fhere[dataset][:],fhere.attrs['TIME'],fhere.attrs['DT']])

    def fu(n):
        plt.figure(figsize=(8, 4))
        plt.plot(xaxis,data[n][0])
        plt.title('time = '+str(data[n][1]))
        if(xlim != [-1,-1]):
            plt.xlim(xlim)
        if(ylim != [-1,-1]):
            plt.ylim(ylim)
        if(zlim != [-1,-1]):
            plt.clim(zlim)
        return plt

    interact(fu,n=(0,len(data)-1))


def phaseinteract(rundir='',dataset='p1x1',species='electrons',
    xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[]):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'PHA', dataset, species)
    files = sorted(os.listdir(odir))

    data = []
    for i in range(len(files)):
        fhere = h5py.File(os.path.join(odir,files[i]), 'r')
        data.append([fhere[dataset][:,:],fhere.attrs['TIME'],fhere.attrs['DT']])
        xaxis = fhere['AXIS/AXIS1'][:]
        yaxis = fhere['AXIS/AXIS2'][:]

    def fu(n):
        plt.figure(figsize=(8, 4))
        plt.imshow(data[n][0],
               extent=[xaxis[0], xaxis[1], yaxis[0], yaxis[1]],
               aspect='auto')
        plt.title('time = '+str(data[n][1]))
        plt.colorbar()
        if(xlim != [-1,-1]):
            plt.xlim(xlim)
        if(ylim != [-1,-1]):
            plt.ylim(ylim)
        if(zlim != [-1,-1]):
            plt.clim(zlim)
        return plt

    interact(fu,n=(0,len(data)-1))


def xtplot(rundir='',dataset='e3',xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[]):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'FLD', dataset)
    files = sorted(os.listdir(odir))

    fhere = h5py.File(os.path.join(odir, files[0]), 'r')
    xaxismin = fhere['AXIS']['AXIS1'][0]
    xaxismax = fhere['AXIS']['AXIS1'][1]
    taxismin = fhere.attrs['TIME']

    fhere = h5py.File(os.path.join(odir, files[len(files)-1]), 'r')
    taxismax = fhere.attrs['TIME']

    nx = len(fhere[dataset][:])
    dx = (xaxismax-xaxismin)/nx
    nt = len(files)

    # fhere = h5py.File(odir+files[len(files)-1], 'r')
    fhere2 = h5py.File(os.path.join(odir, files[len(files)-2]), 'r')
    dt = fhere.attrs['TIME']-fhere2.attrs['TIME']
    # print(dt)

    if(plotdata == []):
        data = []
        for f in files:
            fname = os.path.join(odir, f)
            fhere = h5py.File(fname, 'r')
            data.append([fhere[dataset][:],fhere.attrs['TIME'],fhere.attrs['DT']])
            # xaxis = fhere['AXIS/AXIS1'][:]
            # xaxiselems = [xaxis[0]+n*data[0][2][0] for n in range(len(data[0][0]))]

        plotdata = np.zeros( (len(data),len(data[0][0])) )
        taxis = []
        for i in range(len(data)):
            plotdata[i,:] = data[i][0]
            taxis.append

    plt.figure(figsize=(8, 5))
    plt.imshow(plotdata,
               origin='lower',
               aspect='auto',
              extent=[xaxismin, xaxismax, taxismin, taxismax],
              cmap="nipy_spectral")
    if(xlim != [-1,-1]):
        plt.xlim(xlim)
    if(ylim != [-1,-1]):
        plt.ylim(ylim)
    if(zlim != [-1,-1]):
        plt.clim(zlim)

    plt.colorbar(orientation='vertical')
    plt.title('Time vs Space')
    #plt.xlabel('$x_1 [c/\omega_p]$')
    #plt.ylabel('$p_1 [m_ec]$')

    plt.show()

    return plotdata


def wkplot(rundir='',dataset='e3',klim=[-1,-1],wlim=[-1,-1],zlim=[-1,-1],
    plotdata=[]):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'FLD', dataset)
    files = sorted(os.listdir(odir))

    fhere = h5py.File(os.path.join(odir, files[0]), 'r')
    xaxismin = fhere['AXIS']['AXIS1'][0]
    xaxismax = fhere['AXIS']['AXIS1'][1]
    taxismin = fhere.attrs['TIME']

    fhere = h5py.File(os.path.join(odir, files[len(files)-1]), 'r')
    taxismax = fhere.attrs['TIME']

    nx = len(fhere[dataset][:])
    dx = (xaxismax-xaxismin)/nx
    nt = len(files)

    # fhere = h5py.File(odir+files[len(files)-1], 'r')
    fhere2 = h5py.File(os.path.join(odir, files[len(files)-2]), 'r')
    dt = fhere.attrs['TIME']-fhere2.attrs['TIME']
    # print(dt)

    kaxis = np.fft.fftfreq(nx, d=dx) * 2*np.pi
    waxis = np.fft.fftfreq(nt, d=dt) * 2*np.pi

    if(plotdata == []):

        data = []
        for f in files:
            fname = os.path.join(odir, f)
            fhere = h5py.File(fname, 'r')
            data.append([fhere[dataset][:],fhere.attrs['TIME'],fhere.attrs['DT']])
            # xaxis = fhere['AXIS/AXIS1'][:]
            # xaxiselems = [xaxis[0]+n*data[0][2][0] for n in range(len(data[0][0]))]

        a = np.zeros( (len(data),len(data[0][0])) )
        taxis = []
        for i in range(len(data)):
            a[i,:] = data[i][0]
            taxis.append

        plotdata = np.fliplr(np.fft.fftshift(np.fft.fft2(a)))
        plotdata = np.log(np.abs(plotdata.real))


    fig = plt.figure(figsize=(8, 8))
    plt.imshow(plotdata,
              origin='lower',
               aspect='auto',
              extent=[ min(kaxis), max(kaxis),
                     min(waxis), max(waxis) ],
              cmap="nipy_spectral")

    plt.colorbar(orientation='vertical')

    if(klim != [-1,-1]):
        plt.xlim(klim)
    if(wlim != [-1,-1]):
        plt.ylim(wlim)
    if(zlim != [-1,-1]):
        plt.clim(zlim)

    #fig.show()

    return plotdata


# def modfig(oldfig,klim=[-1,-1],wlim=[-1,-1],zlim=[-1,-1],):
#     #fig = oldfig #plt.figure(figsize=(8, 8))
#     #ax = oldfig.add_subplot(111)
#     ax = oldfig.get_axes()
#
#     if(klim != [-1,-1]):
#         ax.xlim(self,klim)
#     if(wlim != [-1,-1]):
#         ax.ylim(wlim)
#     if(zlim != [-1,-1]):
#         ax.clim(vmin=zlim[0],vmax=zlim[1])
#
#     #oldfig
#     IPython.display.display(oldfig)
#     return oldfig


# ROMAN'S FUNCTIONS
def x(n, one_0 = 10, one_D = 790, n_peak = 2):
    # returns position, x, given density, n
#    one_0 = 10
#    one_D = 790
#    n_peak = 2
    x = one_0 + n/n_peak * one_D
    return x

def k_xm(w):
    # xmode dispersion relation
    c = 1.0
    w_p = 1.0                         # plamsa frequency
    w_c = 0.7                      # cyclotron freq
    w_0 = 1.0
#     k = np.sqrt((w_p**2/c**2) * ( (w/w_p)**2 - ((w/w_p)**2 - 1) 
#                  / ((w/w_p)**2 - (1 + (w_c/w_p)**2) ) ))
    w_H = np.sqrt(w_p**2 + w_c**2)
    
    k = w**2/c**2 * (1 - (w_p**2/w**2) * (w**2 - w_p**2) / (w**2 - w_H**2) )
    return k

def gen_path(rundir, plot_or):
    PATH = os.getcwd() + '/' + rundir
    if (plot_or==1):
        PATH += '/e1.h5'
    elif (plot_or==2):
        PATH += '/e2.h5'
    elif (plot_or==3):
        PATH += '/e3.h5'
    else:
        PATH += '/ions.h5'
    return PATH

def plot_xt_arb(rundir, field='Ex',
            xlim=[None,None], tlim=[None,None]):
    
    # initialize values
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)
    
    if(xlim == [None,None]):
        xlim[0] = hdf5_data.axes[0].axis_min
        xlim[1] = hdf5_data.axes[0].axis_max
    if(tlim == [None,None]):
        tlim[0] = hdf5_data.axes[1].axis_min
        tlim[1] = hdf5_data.axes[1].axis_max


#    y_vals = np.arange(hdf5_data.axes[1].axis_min, hdf5_data.axes[1].axis_max, 1)
#    x_vals = np.full(len(y_vals), x(n_L, one_0=one_0, one_D=one_D, n_peak=n_peak))
#    x_vals2 = np.full(len(y_vals), x(n_R, one_0=one_0, one_D=one_D, n_peak=n_peak))
#    x_vals3 = np.full(len(y_vals), x(w_0*w_0, one_0=one_0, one_D=one_D, n_peak=n_peak))
#    x_vals4 = np.full(len(y_vals), x(w_0**2 - b0_mag**2, one_0=one_0, one_D=one_D, n_peak=n_peak))

    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data )
    plt.title(field + ' x-t space' + field )
    plt.xlabel('x')
    plt.ylabel('t')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(tlim[0],tlim[1])  
    
    plt.show()


def plot_xt(rundir, TITLE='', b0_mag=0.0, w_0 = 1.0, one_0 = 10, one_D= 790, n_peak = 2, plot_or=3, show_theory=False,
            xlim=[None,None], tlim=[None,None], **kwargs):
    
    # initialize values
    PATH = gen_path(rundir, plot_or)
    hdf5_data = read_hdf(PATH)
    
    if(xlim == [None,None]):
        xlim[0] = hdf5_data.axes[0].axis_min
        xlim[1] = hdf5_data.axes[0].axis_max
    if(tlim == [None,None]):
        tlim[0] = hdf5_data.axes[1].axis_min
        tlim[1] = hdf5_data.axes[1].axis_max

#    w_0 = 1.0
    n_L = w_0**2 + w_0*b0_mag
    n_R = w_0**2 - w_0*b0_mag

    y_vals = np.arange(hdf5_data.axes[1].axis_min, hdf5_data.axes[1].axis_max, 1)
    x_vals = np.full(len(y_vals), x(n_L, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals2 = np.full(len(y_vals), x(n_R, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals3 = np.full(len(y_vals), x(w_0*w_0, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals4 = np.full(len(y_vals), x(w_0**2 - b0_mag**2, one_0=one_0, one_D=one_D, n_peak=n_peak))

    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data, **kwargs)
    plt.title(TITLE + ' x-t space' + ' e' + str(plot_or))
    plt.xlabel('x')
    plt.ylabel('t')
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(tlim[0],tlim[1])  
    if (show_theory==True):
        plt.plot(x_vals, y_vals, 'c--', label='$\omega_L$ x-cutoff') #L-cutoff
        plt.plot(x_vals2, y_vals, 'b--', label='$\omega_R$ x-cutoff')#R-cutoff
        plt.plot(x_vals3, y_vals, 'r--', label='$\omega_p$')
        plt.plot(x_vals4, y_vals, 'g--', label='$\omega_H$')
        plt.legend(loc=4)
    plt.show()
    
def plot_tx(rundir, TITLE='', b0_mag=0.0, plot_or=3, show_theory=False,
            xlim=[None,None], tlim=[None,None], show_cutoff=False, w_0 = 1.0, one_0 = 10, one_D= 790, n_peak = 2, **kwargs):

    # initialize values
    PATH = gen_path(rundir, plot_or)
    hdf5_data = read_hdf(PATH)

    if(xlim == [None,None]):
        xlim[0] = hdf5_data.axes[0].axis_min
        xlim[1] = hdf5_data.axes[0].axis_max
    if(tlim == [None,None]):
        tlim[0] = hdf5_data.axes[1].axis_min
        tlim[1] = hdf5_data.axes[1].axis_max

#    w_0 = 1.0
    n_L = w_0**2 + w_0*b0_mag
    n_R = w_0**2 - w_0*b0_mag

    y_vals = np.arange(hdf5_data.axes[1].axis_min, hdf5_data.axes[1].axis_max, 1)
    x_vals = np.full(len(y_vals), x(n_L, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals2 = np.full(len(y_vals), x(n_R, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals3 = np.full(len(y_vals), x(w_0*w_0, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals4 = np.full(len(y_vals), x(w_0**2 - b0_mag**2, one_0=one_0, one_D=one_D, n_peak=n_peak))
    x_vals5 = np.full(len(y_vals), 30.0)

    # create figure
    plt.figure(figsize=(8,5))
    plotmetranspose(hdf5_data, np.transpose(hdf5_data.data), **kwargs)
    plt.title(TITLE + ' t-x space' + ' e' + str(plot_or))
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xlim(tlim[0],tlim[1])
    plt.ylim(xlim[0],xlim[1])
    if (show_theory==True):
        plt.plot(y_vals, x_vals, 'c--', label='$\omega_L$ x-cutoff') #L-cutoff
        plt.plot(y_vals, x_vals2, 'b--', label='$\omega_R$ x-cutoff')#R-cutoff
        plt.plot(y_vals, x_vals3, 'r--', label='$\omega_p$')
        plt.plot(y_vals, x_vals4, 'g--', label='$\omega_H$')
        plt.legend(loc=1)
    if (show_cutoff==True):
        plt.plot(y_vals, x_vals5,'b', label='')
    plt.show()
    

def plot_log_xt(PATH, TITLE):
    # initialize values
    hdf5_data = read_hdf(PATH)

    # log the hdf5 data
    s = 1e-3  #sensitivity
    for i in range(hdf5_data.data.shape[0]):
        for j in range(hdf5_data.data.shape[1]):
            if abs(hdf5_data.data[i,j]) > s:
                hdf5_data.data[i,j] = np.log(abs(hdf5_data.data[i,j]))
            else:
                hdf5_data.data[i,j] = 0

    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data)
    plt.title(TITLE + ' x-t space log plot')
    plt.xlabel('x')
    plt.ylabel('t')
    # plt.legend(loc=0)
    plt.show()

    
def plot_wk(rundir, TITLE='', vth=0.1, b0_mag=0.0, plot_or=1, show_theory=False, 
            wlim=[None,None], klim=[None,None], debye=False, **kwargs):
  
    # initialize values
    PATH = gen_path(rundir, plot_or)
    hdf5_data = read_hdf(PATH)
    hdf5_data = FFT_hdf5(hdf5_data)   # FFT the data (x-t -> w-k)
    if (debye==True):
        hdf5_data.axes[0].axis_min *= vth
        hdf5_data.axes[0].axis_max *= vth

    if(wlim == [None,None]):
        #nt = hdf5_data.shape[0]
        #dt = hdf5_data.axes[1].axis_max/(hdf5_data.shape[0]-1)
        #waxis = np.fft.fftfreq(nt, d=dt) * 2*np.pi
        wlim[0] = hdf5_data.axes[1].axis_min
        wlim[1] = hdf5_data.axes[1].axis_max
    if(klim == [None,None]):
        #nx = hdf5_data.shape[0]
        #dx = hdf5_data.axes[0].axis_max/hdf5_data.shape[1]
        #kaxis = np.fft.fftfreq(nx, d=dx) * 2*np.pi
        klim[0] = hdf5_data.axes[0].axis_min
        klim[1] = hdf5_data.axes[0].axis_max
        
    w_p = 1.0                         # plamsa frequency
    w_c = b0_mag                      # cyclotron freq
    w_0 = 1.0
    w_H = np.sqrt(w_p**2 + w_c**2)

    N = 100
    dx = float(klim[1])/N
    kvals = np.arange(0, klim[1]+.01, dx)

    if (b0_mag==0 and plot_or==1):
        if (debye==True):
            wvals = np.sqrt(w_p**2 + 3 * vth**2 * (kvals/vth)**2)
        else:
            wvals = np.sqrt(w_p**2 + 3 * vth**2 * (kvals)**2)
    else:
        if (debye==True):
            wvals = np.sqrt(w_p**2 + (kvals/vth)**2)
        else:
            wvals = np.sqrt(w_p**2 + kvals**2)

    wR = np.array([0.5 * ( w_c + np.sqrt(w_c**2 + 4 * w_p**2))
                   for i in np.arange(len(kvals))])            # right-handed cutoff
    wL = np.array([0.5 * (-w_c + np.sqrt(w_c**2 + 4 * w_p**2))
                   for i in np.arange(len(kvals))])            # left-handed cutoff
    w_H_vals = np.array([w_H for i in np.arange(len(kvals))])            # hybrid frequency
    w_p_vals = np.array([w_p for i in np.arange(len(kvals))])
    w_cvals = np.array([w_c for i in np.arange(len(kvals))])
    
    #arrays for xmode theory curve
    wvals_xm = np.arange(0.1, 5, dx/100.0)
    kvals_xm = (wvals_xm**2 * (1 - (w_p**2/wvals_xm**2) * (wvals_xm**2 - w_p**2) / (wvals_xm**2 - w_H**2) ))
    kvals_xm = np.where(kvals_xm > 0, kvals_xm, 0)
    kvals_xm = np.sqrt(kvals_xm)
        
    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data, **kwargs)
    plt.title(TITLE + ' w-k space' + ' e' + str(plot_or))
    if (debye==True):
        plt.xlabel('k  [$\lambda_D$]')
    else:
        plt.xlabel('k  [$\omega_{pe}$/c]')
    plt.ylabel('$\omega$  [$\omega_{pe}$]')
    plt.xlim(klim[0],klim[1])
    plt.ylim(wlim[0],wlim[1])   
    if (show_theory==True):
        if (b0_mag!=0):
            # for i in range(1,10):
            #     plt.plot(kvals, i*w_cvals, 'w--', label='')
            if (plot_or==2 or plot_or==1):
                # xmode
                plt.plot(kvals_xm, wvals_xm, 'fuchsia', label='x-wave dispersion relation')
                plt.plot(kvals, wR, 'r--', label='$\omega_R$ cutoff')
                plt.plot(kvals, wL, 'w--', label='$\omega_L$ cutoff') 
                plt.plot(kvals, w_p_vals, 'r:', label='$\omega_p$') 
                plt.plot(kvals, w_H_vals, 'w:', label='$\omega_H$, hybrid frequency')
            elif (plot_or==3):
                # omode
                plt.plot(kvals, wvals,'fuchsia', label='o-wave dispersion relation')
#                 plt.plot(kvals, wR, 'r--', label='$\omega_R$, right-handed cutoff')
#                 plt.plot(kvals, wL, 'w--', label='$\omega_L$, left-handed cutoff') 
            plt.legend(loc=0)
        else:
            plt.plot(kvals,wvals,'red')   
    plt.show()
    

def plot_wk_rl(rundir, TITLE='', vth=0.1, b0_mag=0.0, plot_or=1, show_theory=False, 
            wlim=[None,None], klim=[None,None], debye=False, **kwargs):
  
    # initialize values
    PATH = gen_path(rundir, plot_or)
    hdf5_data = read_hdf(PATH)
    hdf5_data = FFT_hdf5(hdf5_data)   # FFT the data (x-t -> w-k)
    if (debye==True):
        hdf5_data.axes[0].axis_min *= vth
        hdf5_data.axes[0].axis_max *= vth

    if(wlim == [None,None]):
        #nt = hdf5_data.shape[0]
        #dt = hdf5_data.axes[1].axis_max/(hdf5_data.shape[0]-1)
        #waxis = np.fft.fftfreq(nt, d=dt) * 2*np.pi
        wlim[0] = hdf5_data.axes[1].axis_min
        wlim[1] = hdf5_data.axes[1].axis_max
    if(klim == [None,None]):
        #nx = hdf5_data.shape[0]
        #dx = hdf5_data.axes[0].axis_max/hdf5_data.shape[1]
        #kaxis = np.fft.fftfreq(nx, d=dx) * 2*np.pi
        klim[0] = hdf5_data.axes[0].axis_min
        klim[1] = hdf5_data.axes[0].axis_max
        
    w_p = 1.0                         # plamsa frequency
    w_c = b0_mag                      # cyclotron freq
#    w_0 = 1.0
    w_H = np.sqrt(w_p**2 + w_c**2)

    N = 100
    dx = float(klim[1])/N
    kvals = np.arange(0, klim[1]+.01, dx)

    if (b0_mag==0 and plot_or==1):
        if (debye==True):
            wvals = np.sqrt(w_p**2 + 3 * vth**2 * (kvals/vth)**2)
        else:
            wvals = np.sqrt(w_p**2 + 3 * vth**2 * (kvals)**2)
    else:
        if (debye==True):
            wvals = np.sqrt(w_p**2 + (kvals/vth)**2)
        else:
            wvals = np.sqrt(w_p**2 + kvals**2)

    wR = np.array([0.5 * ( w_c + np.sqrt(w_c**2 + 4 * w_p**2))
                   for i in np.arange(len(kvals))])            # right-handed cutoff
    wL = np.array([0.5 * (-w_c + np.sqrt(w_c**2 + 4 * w_p**2))
                   for i in np.arange(len(kvals))])            # left-handed cutoff
    w_H_vals = np.array([w_H for i in np.arange(len(kvals))])            # hybrid frequency
    w_p_vals = np.array([w_p for i in np.arange(len(kvals))])
    w_cvals = np.array([w_c for i in np.arange(len(kvals))])
    
    #arrays for rmode theory curve
    wvals_rm = np.arange(0.1, 5, dx/100.0)
    #kvals_xm = (wvals_xm**2 * (1 - (w_p**2/wvals_xm**2) * (wvals_xm**2 - w_p**2) / (wvals_xm**2 - w_H**2) ))
    kvals_rm = (wvals_rm**2 * (1 - (w_p**2/wvals_rm**2) / (1 - w_c / wvals_rm) ))
    kvals_rm = np.where(kvals_rm > 0, kvals_rm, 0)
    kvals_rm = np.sqrt(kvals_rm)
    #arrays for lmode theory curve
    wvals_lm = np.arange(0.1, 5, dx/100.0)
    #kvals_xm = (wvals_xm**2 * (1 - (w_p**2/wvals_xm**2) * (wvals_xm**2 - w_p**2) / (wvals_xm**2 - w_H**2) ))
    kvals_lm = (wvals_rm**2 * (1 - (w_p**2/wvals_lm**2) / (1 + w_c / wvals_lm) ))
    kvals_lm = np.where(kvals_lm > 0, kvals_lm, 0)
    kvals_lm = np.sqrt(kvals_lm)
        
    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data, **kwargs)
    plt.title(TITLE + ' w-k space' + ' e' + str(plot_or))
    if (debye==True):
        plt.xlabel('k  [$\lambda_D$]')
    else:
        plt.xlabel('k  [$\omega_{pe}$/c]')
    plt.ylabel('$\omega$  [$\omega_{pe}$]')
    plt.xlim(klim[0],klim[1])
    plt.ylim(wlim[0],wlim[1])   
    if (show_theory==True):
        if (b0_mag!=0):
            # for i in range(1,10):
            #     plt.plot(kvals, i*w_cvals, 'w--', label='')
            if (plot_or==2 or plot_or==3):
                # xmode
                plt.plot(kvals_rm, wvals_rm, 'white', label='R-wave dispersion relation')
                plt.plot(kvals_lm, wvals_lm, 'red', label='L-wave dispersion relation')
                plt.plot(kvals, wR, 'w--', label='$\omega_R$ cutoff')
                plt.plot(kvals, wL, 'r--', label='$\omega_L$ cutoff') 
                plt.plot(kvals, w_cvals, 'blue', label='$\omega_c$') 
                #plt.plot(kvals, w_p_vals, 'r:', label='$\omega_p$') 
                #plt.plot(kvals, w_H_vals, 'w:', label='$\omega_H$, hybrid frequency')
            elif (plot_or==1):
                # omode
                wvals = w_p + 0 * kvals
                plt.plot(kvals, wvals,'red', label='$\omega_p$')
#                 plt.plot(kvals, wR, 'r--', label='$\omega_R$, right-handed cutoff')
#                 plt.plot(kvals, wL, 'w--', label='$\omega_L$, left-handed cutoff') 
            plt.legend(loc=0)
            
    plt.show()

    
def plot_wk_iaw(rundir, TITLE, show_theory=False, background=0.0, wlim=3, klim=5):
    
    # initialize values
    PATH = os.getcwd() + '/' + rundir + '/ions.h5'
    hdf5_data = read_hdf(PATH)
    if (background!=0.0):
        hdf5_data.data = hdf5_data.data-background
    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)

    c_s = 0.01                              # sound speed
    w_pi = 0.1                              # plasma ion freq

    N = 100
    dx = float(klim)/N
    kvals = np.arange(0, klim+.01, dx)
    wvals = kvals * c_s

    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data)
    if (plot_or !=4):
        plt.title(TITLE + ' w-k space' + ' e' + str(plot_or))
    else:
        plt.title(TITLE + ' w-k space' + ' ion density' )
    plt.xlabel('k  [$\omega_{pe}$/c]')
    plt.ylabel('$\omega$  [$\omega_{pe}$]')
    plt.xlim(0,klim)
    plt.ylim(0,wlim)
    if (show_theory==True):
        plt.plot(kvals, wvals,'b', label='')
        plt.legend(loc=0)
    plt.show()
    
def plot_wk_arb(rundir, field, TITLE, background=0.0, wlim=3, klim=5):
    
    # initialize values
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)
    if (background!=0.0):
        hdf5_data.data = hdf5_data.data-background
    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)

    c_s = 0.01                              # sound speed
    w_pi = 0.1                              # plasma ion freq

    N = 100
    dx = float(klim)/N
    kvals = np.arange(0, klim+.01, dx)
    wvals = kvals * c_s

    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data)
    
    plt.title(TITLE + ' w-k space' +  TITLE)
    
    plt.xlabel('k  [$1/ \Delta x$]')
    plt.ylabel('$\omega$  [$\omega_{pe}$]')
    plt.xlim(0,klim)
    plt.ylim(0,wlim)
    plt.show()
    

def get_ratio(PATH1, PATH2):
    #Function gets ratio of hdf52 and hdf51 in w-k space
    hdf51 = read_hdf(PATH1)
    hdf52 = read_hdf(PATH2)

    hdf51 = FFT_hdf5(hdf51)   # FFT the data (x-t -> w-k)
    hdf52 = FFT_hdf5(hdf52)   # FFT the data (x-t -> w-k)

    s = 20  #sensitivity
    for i in range(hdf51.shape[0]):
       for j in range(hdf51.shape[1]):
           hdf51.data[i,j] = abs(hdf51.data[i,j]/hdf52.data[i,j])
           if (hdf51.data[i,j]>s):
               hdf51.data[i,j] = 0.0

    return hdf51


def plot_mode_hist(hdf5):
    # plot mode history for a couple of modes
    plt.figure()
    plt.semilogy(hdf5.data[:,4])
    plt.semilogy(hdf5.data[:,8])
    plt.show()
    
def phaseinteract_upic(rundir='',
   xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
   plotdata=[]):

   workdir = os.getcwd()
   workdir = os.path.join(workdir, rundir)

   odir = os.path.join(workdir, 'DIAG/Vx_x')
   files = sorted(os.listdir(odir))

   data = []
   for i in range(len(files)):
       fhere = h5py.File(os.path.join(odir,files[i]), 'r')
       data.append([fhere['Phase Space - vx vs. x'][:,:],fhere.attrs['TIME'],fhere.attrs['DT']])
       xaxis = fhere['AXIS/AXIS1'][:]
       yaxis = fhere['AXIS/AXIS2'][:]

   def fu(n):
       plt.figure(figsize=(8, 4))
       plt.imshow(np.log(np.abs(data[n][0]+0.001)),cmap='OrRd',origin='lower',
              extent=[xaxis[0], xaxis[1], yaxis[0], yaxis[1]],
              aspect='auto')
       plt.title('time = '+str(data[n][1]))
       plt.colorbar()
       if(xlim != [-1,-1]):
           plt.xlim(xlim)
       if(ylim != [-1,-1]):
           plt.ylim(ylim)
       if(zlim != [-1,-1]):
           plt.clim(zlim)
       return plt

   interact(fu,n=(0,len(data)-1))
