import os
import shutil
import subprocess
import IPython.display
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ipywidgets import interact, fixed
from h5_utilities import *
from analysis import *
from scipy.optimize import fsolve

from scipy import special
import cmath
import mpmath

# FEB 2019
# Now using Han Wen's library for I/O, Feb 2019
import osh5io
import osh5def
import osh5vis

import osh5utils

import matplotlib.colors as colors
import ipywidgets

import ipywidgets as widgets
#



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
            print(path, end='')

    def combine_h5_iaw_2d():
        in_file = workdir + '/DIAG/IDen/'
        out_file = workdir + '/ions.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_2d.py", in_file, out_file]):
            print(path, end='')
            IPython.display.clear_output(wait=True)

    workdir = os.getcwd()
    if os.path.isfile(workdir+'/upic-es.out'):
        localexec = workdir+'/upic-es.out'
    else:
        localexec = False
    sysexec = '/usr/local/beps/upic-es.out'
    workdir += '/' + rundir
    print(workdir)

    if(not os.path.isdir(workdir)):
       os.mkdir(workdir)
    if(rundir != ''):
        shutil.copyfile(inputfile,workdir+'/pinput2')
    
    os.chdir(workdir)


    # run the upic-es executable
    print('running upic-es.out ...')
    if localexec:
        for path in execute([localexec]):
            pass
    else:
        for path in execute([sysexec]):
            pass
    IPython.display.clear_output(wait=True)

    # run the combine script on electric field data
    print('combining Ex files')
    combine_h5_2d('DIAG', 'Ex')
    combine_h5_2d('DIAG', 'Ey')
    combine_h5_2d('DIAG', 'pot')
    print('combine_h5_2d completed normally')
    
    # run combine on iaw data if present
    if (os.path.isdir(workdir + '/DIAG/IDen/')):
        combine_h5_iaw_2d()
        print('combine_h5_iaw completed normally')


    IPython.display.clear_output(wait=True)
    print('run_upic_es completed normally')

    os.chdir('../')
    
    return
    

def runosiris(rundir='',inputfile='osiris-input.txt',print_out='yes',combine='yes'):

    def combine_h5_1d(ex):
        in_file = workdir + '/MS/FLD/' + ex + '/'
        out_file = workdir + '/' + ex + '.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_1d.py", in_file, out_file]):
            if print_out == 'yes':
                IPython.display.clear_output(wait=True)
#            print(path, end='')

    def combine_h5_iaw_1d():
        in_file = workdir + '/MS/DENSITY/ions/charge/'
        out_file = workdir + '/ions.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_1d.py", in_file, out_file]):
            if print_out == 'yes':
                IPython.display.clear_output(wait=True)
#            print(path, end='')

    workdir = os.getcwd()
    if os.path.isfile(workdir+'/osiris-1D.e'):
        localexec = workdir+'/osiris-1D.e'
    else:
        localexec = False
    sysexec = '/usr/local/osiris/osiris-1D.e'
    workdir += '/' + rundir
    if print_out == 'yes':
        print(workdir)

    # run osiris-1D.e executable
    if(not os.path.isdir(workdir)):
        os.mkdir(workdir)
    else:
        shutil.rmtree(workdir)
        os.mkdir(workdir)
    if(rundir != ''):
#        shutil.copyfile('osiris-1D.e',workdir+'/osiris-1D.e')
        shutil.copyfile(inputfile,workdir+'/osiris-input.txt')
    waittick = 0

    if localexec:
        for path in execute([localexec,"-w",workdir,"osiris-input.txt"]):
            if print_out == 'yes':
                waittick += 1
                if(waittick == 100):
                    IPython.display.clear_output(wait=True)
                    waittick = 0
                    print(path, end='')
    else:
        for path in execute([sysexec,"-w",workdir,"osiris-input.txt"]):
            if print_out == 'yes':
                waittick += 1
                if(waittick == 100):
                    IPython.display.clear_output(wait=True)
                    waittick = 0
                    print(path, end='')

    # run combine_h5_util_1d.py script for e1/, e2/, e3/ (and iaw if applicable)

    if print_out == 'yes':
        print('combining E1 files')
    if combine == 'yes':
        combine_h5_1d('e1')
    if print_out == 'yes':
        print('combining E2 files')
    if combine == 'yes':
        combine_h5_1d('e2')
    if print_out == 'yes':
        print('combining E3 files')
    if combine == 'yes':
        combine_h5_1d('e3')

    # run combine on iaw data if present
    if (os.path.isdir(workdir+'/MS/DENSITY/ions/charge')):
        if print_out == 'yes':
            print('combining IAW files')
        if combine == 'yes':
            combine_h5_iaw_1d()

    if print_out == 'yes':
        IPython.display.clear_output(wait=True)
        print('runosiris completed normally')

    return


def runosiris_2d(rundir='',inputfile='osiris-input.txt'):

    def combine_h5_2d(ex):
        in_file = workdir + '/MS/FLD/' + ex + '/'
        out_file = workdir + '/' + ex + '.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_2d_true.py", in_file, out_file]):
            IPython.display.clear_output(wait=True)
#            print(path, end='')

    def combine_h5_iaw_2d():
        in_file = workdir + '/MS/DENSITY/ions/charge/'
        out_file = workdir + '/ions.h5'
        for path in execute(["python", "/usr/local/osiris/combine_h5_util_2d_true.py", in_file, out_file]):
            IPython.display.clear_output(wait=True)
#            print(path, end='')

    workdir = os.getcwd()
    if os.path.isfile(workdir+'/osiris-2D.e'):
        localexec = workdir+'/osiris-2D.e'
    else:
        localexec = False
    sysexec = '/usr/local/osiris/osiris-2D.e'
    workdir += '/' + rundir
    print(workdir)

    # run osiris-2D.e executable
    if(not os.path.isdir(workdir)):
       os.mkdir(workdir)
    if(rundir != ''):
#        shutil.copyfile('osiris-2D.e',workdir+'/osiris-2D.e')
        shutil.copyfile(inputfile,workdir+'/osiris-input.txt')
    waittick = 0
    if localexec:
        for path in execute([localexec,"-w",workdir,"osiris-input.txt"]):
            waittick += 1
            if(waittick == 100):
                IPython.display.clear_output(wait=True)
                waittick = 0
                print(path, end='')
    else:
        for path in execute([sysexec,"-w",workdir,"osiris-input.txt"]):
            waittick += 1
            if(waittick == 100):
                IPython.display.clear_output(wait=True)
                waittick = 0
                print(path, end='')

    # run combine_h5_util_1d.py script for e1/, e2/, e3/ (and iaw if applicable)
    print('combining E1 files')
    combine_h5_2d('e1')
    print('combining E2 files')
    combine_h5_2d('e2')
    print('combining E3 files')
    combine_h5_2d('e3')
    print('combining B1 files')
    combine_h5_2d('b1')
    print('combining B2 files')
    combine_h5_2d('b2')
    print('combining B3 files')
    combine_h5_2d('b3')

    # run combine on iaw data if present
    if (os.path.isdir(workdir+'/MS/DENSITY/ions/charge')):
        print('combining IAW files')
        combine_h5_iaw_2d()

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
            xlim=[None,None], tlim=[None,None],plot_show=True):
    
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

# an option not to complete the plot, in case you want to perform the analysis outside
#
    if (plot_show):
        plt.show()
#
#


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
    
def plot_wk_arb(rundir, field, TITLE, background=0.0, wlim=3, klim=5,plot_show=True):
    
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
    plt.figure(figsize=(10,10))
    plotme(hdf5_data)
    
    plt.title(TITLE + ' w-k space' +  TITLE)
    
    plt.xlabel('k  [$1/ \Delta x$]')
    plt.ylabel('$\omega$  [$\omega_{pe}$]')
    plt.xlim(0,klim)
    plt.ylim(0,wlim)
    if (plot_show):
        plt.show()
    
def plot_tk_arb(rundir, field, title='potential', klim=5,tlim=100):

    
    title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'34'}
    # initialize values
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)
    
#    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)
    k_data=np.fft.fft(hdf5_data.data,axis=1)
    hdf5_data.data=np.abs(k_data)
    
    hdf5_data.axes[0].axis_max=2.0*3.1415926
 

#    N = 100
#    dx = float(klim)/N
#    kvals = np.arange(0, klim+.01, dx)
#    wvals = kvals * c_s

    # create figure
    plt.figure(figsize=(10,10))
    plotme(hdf5_data)
    
    plt.title(title + ' t-k space' )
    
    plt.xlabel('k  [$1/ \Delta x$]',**axis_font)
    plt.ylabel(' Time  [$1/ \omega_{pe}$]',**axis_font)
    plt.xlim(0,klim)
    plt.ylim(0,tlim)
    plt.show()

def wk_upic_iaw(rundir, field, TITLE='', background=0.0, wlim=[None,None],
                klim=[None,None], show_theory=True, **kwargs):

    # initialize values
    PATH = os.getcwd() + '/' + rundir + '/' + field + '.h5'
    hdf5_data = read_hdf(PATH)
    if (background!=0.0):
        hdf5_data.data = hdf5_data.data-background
    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)

    if(wlim == [None,None]):
        wlim[0] = hdf5_data.axes[1].axis_min
        wlim[1] = hdf5_data.axes[1].axis_max
    if(klim == [None,None]):
        klim[0] = hdf5_data.axes[0].axis_min
        klim[1] = hdf5_data.axes[0].axis_max

    # create fluid theory dispersion relation
    def w(k, vtx=1.0, rmass=100.0):
        c_s = vtx/np.sqrt(rmass)  # VTX/sqrt(RMASS) in input deck
        k_DE = 1/vtx
        w = k*c_s/np.sqrt(1+(k/k_DE)**2)
        return w

    ks = np.linspace(klim[0],klim[1],100*(klim[1]-klim[0]))
    ws = w(ks, **kwargs)

    # create figure
    plt.figure(figsize=(8,5))
    plotme(hdf5_data)
    plt.title(TITLE + ' $\omega$-k space' +  TITLE)
    plt.xlabel('k  [$1/ \Delta x$]')
    plt.ylabel('$\omega$  [$\omega_{pe}$]')
    plt.xlim(klim[0],klim[1])
    plt.ylim(wlim[0],wlim[1])
    if (show_theory==True):
        plt.plot(ks,ws)
        plt.show()


def plot_tk_2stream(rundir, field, klim=5,tlim=100,v0=1):

    
    title_font = { 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
    axis_font = { 'size':'34'}
    # initialize values
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)
    
#    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)
    k_data=np.fft.fft(hdf5_data.data,axis=1)
    hdf5_data.data=np.abs(k_data)
    
    hdf5_data.axes[0].axis_max=2.0*3.1415926*v0
 

#    N = 100
#    dx = float(klim)/N
#    kvals = np.arange(0, klim+.01, dx)
#    wvals = kvals * c_s
    N=100
    dt = float(tlim)/N
    tvals=np.arange(0,tlim,dt)
    kvals=np.zeros(N)
    kpeak_vals=np.zeros(N)
    for i in range(0,N):
        kvals[i]=np.sqrt(2)
        kpeak_vals[i]=0.85
        
   
    # create figure
    plt.figure(figsize=(10,10))
    plotme(hdf5_data)
    plt.plot(kvals,tvals,'b--',label='Instability Boundary')
    plt.plot(kpeak_vals,tvals,'r--',label='Peak Location')
    
    plt.title(field + ' t-k space' )
    
    plt.xlabel(' α ',**axis_font)
    plt.ylabel(' Time  [$1/ \omega_{pe}$]',**axis_font)
    plt.xlim(0,klim)
    plt.ylim(0,tlim)
    plt.legend()
    plt.show()
 

def plot_tk_landau_theory(rundir, field, modeno=22,tlim=100, theory1=0.01, theory2=0.01,init_amplitude=1e-5):

    
    
    
    title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'34'}
    # initialize values
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)
    
#    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)
    k_data=np.fft.fft(hdf5_data.data,axis=1)
    hdf5_data.data=np.abs(k_data)

    nx=hdf5_data.data.shape[1]
    nt=hdf5_data.data.shape[0]
    taxis=np.linspace(0,hdf5_data.axes[1].axis_max,nt)
    deltak=2.0*3.1415926/nx
 
    
#    N = 100
#    dx = float(klim)/N
#    kvals = np.arange(0, klim+.01, dx)
#    wvals = kvals * c_s
    N=100
#    dt = float(tlim)/N
#    tvals=np.arange(0,tlim,dt)
#    kvals=np.zeros(N)
#    for i in range(0,N):
#        kvals[i]=np.sqrt(2)
        
   
    # create figure
    plt.figure(figsize=(8,6))
    
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 32
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title   

    # plt.subplot(nplots,1,imode)
    landau_theory1=np.zeros(nt)
    landau_theory2=np.zeros(nt)
    # growth_rate=tstream_root_minus_i(deltak*imode,v0,1.0)
    for it in range(0,nt):
        landau_theory1[it]=init_amplitude*np.exp(-theory1*taxis[it])
        landau_theory2[it]=init_amplitude*np.exp(-theory2*taxis[it])

#plt.figure(figsize=(12,8))
    plt.semilogy(taxis,hdf5_data.data[:,modeno],label='PIC simulation, mode ='+repr(modeno))
    label1='Exact Root, Damping = {0:.3f} '.format(float(theory1))
    label2='df/dv estimate, Damping = {0:.3f} '.format(float(theory2))
    plt.semilogy(taxis,landau_theory1,'r',label=label1)
    plt.semilogy(taxis,landau_theory2,'g',label=label2)
    plt.ylabel('mode'+repr(modeno))
    plt.xlabel('Time [$1/ \omega_{p}$]')
    plt.legend()
    plt.xlim(0,tlim)


        
#    plt.xlabel(' α ',**axis_font)
#    plt.ylabel(' Time  [$1/ \omega_{pe}$]',**axis_font)
#    plt.xlim(0,klim)
#    plt.ylim(0,tlim)
    plt.show()
    
    
def plot_tk_2stream_theory(rundir, field, modemin=1,modemax=5,tlim=100,v0=1,init_amplitude=1e-5):

    
    
    
    title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
    axis_font = {'fontname':'Arial', 'size':'34'}
    # initialize values
    PATH = os.getcwd() + '/' + rundir +'/'+ field + '.h5'
    hdf5_data = read_hdf(PATH)
    
#    hdf5_data = FFT_hdf5(hdf5_data)         # FFT the data (x-t -> w-k)
    k_data=np.fft.fft(hdf5_data.data,axis=1)
    hdf5_data.data=np.abs(k_data)

    nx=hdf5_data.data.shape[1]
    nt=hdf5_data.data.shape[0]
    taxis=np.linspace(0,hdf5_data.axes[1].axis_max,nt)
    deltak=2.0*3.1415926/nx
    hdf5_data.axes[0].axis_max=2.0*3.1415926*v0
 
    nplots=modemax-modemin+1
    
#    N = 100
#    dx = float(klim)/N
#    kvals = np.arange(0, klim+.01, dx)
#    wvals = kvals * c_s
    N=100
#    dt = float(tlim)/N
#    tvals=np.arange(0,tlim,dt)
#    kvals=np.zeros(N)
#    for i in range(0,N):
#        kvals[i]=np.sqrt(2)
        
   
    # create figure
    plt.figure(figsize=(10,3*nplots))
#    plotme(hdf5_data)
    
#    plt.title(field + ' t-k space' )
    for imode in range(modemin,modemax+1):
        plt.subplot(nplots,1,imode)
        stream_theory=np.zeros(nt)
        growth_rate=tstream_root_minus_i(deltak*imode,v0,1.0)
        for it in range(0,nt):
            stream_theory[it]=init_amplitude*np.exp(growth_rate*taxis[it])

#plt.figure(figsize=(12,8))
        plt.semilogy(taxis,hdf5_data.data[:,imode],label='PIC simulation, mode ='+repr(imode))
        plt.semilogy(taxis,stream_theory,'r',label='theory, growth rate ='+repr(growth_rate))
        plt.ylabel('mode'+repr(imode))
        plt.xlabel('Time [$1/ \omega_{p}$]')
        plt.legend()
        plt.xlim(0,tlim)


        
#    plt.xlabel(' α ',**axis_font)
#    plt.ylabel(' Time  [$1/ \omega_{pe}$]',**axis_font)
#    plt.xlim(0,klim)
#    plt.ylim(0,tlim)
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
    
    
################################################################
##  the functions below are for the streaming instability demos
##  f.s. tsung & k. miller
##  (c) 2018 Regents of The University of California
#################################################################

def tstream_root_plus(k,v0,omegap):
    alpha=k*v0/omegap
    result = omegap*np.sqrt(1+alpha*alpha+np.sqrt(1+4*alpha*alpha))
    return result

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
        
    plt.figure(figsize=(10,10))
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

    
################################################################
##  the functions below are for the buneman instability notebooks
##  f.s. tsung & k. g. miller
##  (c) 2018 Regents of The University of California
#################################################################

def buneman_growth_rate(alphaarray,rmass):
    
    nalpha=alphaarray.shape[0]
    
    alphamin=alphaarray[0]
    alphamax=alphaarray[nalpha-1]
    
    prev_root=complex(0,0)
    
    growth_rate=np.zeros(nalpha)
    
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
        prev_root=complex(new_root.real,new_root.imag)
    
    return growth_rate


    
    

################################################################
##  plasma dispersion functions
##  f.s. tsung 
##  (c) 2018 Regents of The University of California
#################################################################


def zfunc(z):
    a = special.wofz(z)
    a *= np.sqrt(np.pi)*complex(0,1)
    return a


def zprime(z):

## the line below is needed for the root finder, which uses MP (multi-precision) variables 
## instead of real and/or complex, so the first step is to convert the variable "z" from 
## of a type 
    
    arg= complex(z.real,z.imag)
    value= zfunc(arg)
    return(-2*(1+z*value))


def landau(karray):

    nk=karray.shape[0]

    results=np.zeros(nk)
    results_r = np.zeros(nk)

    kmin=karray[0]
    kmax=karray[nk-1]

    if (kmin!=0.0):
        root_trial=complex(1,0)

        for k_val in np.arange(0.01,kmin,0.01):
            def epsilon(omega):
                return 1-0.5*((1.0/k_val)**2)*zprime(omega/(np.sqrt(2)*k_val))
            newroot=mpmath.findroot(epsilon,root_trial,solver='muller')
            root_trial=newroot

        results[0]=newroot.imag
    else:
        results[0]=0.0
        newroot=complex(1,0)
        root_trial=complex(1,0)


    for i_mode in range(1,nk):
        k_val=karray[i_mode]
        def epsilon(omega):
            return 1-0.5*((1.0/k_val)**2)*zprime(omega/(np.sqrt(2)*k_val))
        newroot=mpmath.findroot(epsilon,root_trial,solver='muller')
        root_trial=newroot
        results[i_mode]=newroot.imag
        results_r[i_mode] = newroot.real

    return results, results_r



################################################################
##  Kyle's 2D functions
##  K.G. Miller
##  (c) 2018 Regents of The University of California
#################################################################

def field_2d(rundir='',dataset='e1',time=0,space=-1,
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
    plt.title(dataset+' at t = '+str(fhere.attrs['TIME'][0]))
    plt.xlabel('$x_1 [c/\omega_p]$')
    plt.ylabel('$x_2 [c/\omega_p]$')

    xaxismin = fhere['AXIS']['AXIS1'][0]
    xaxismax = fhere['AXIS']['AXIS1'][1]
    yaxismin = fhere['AXIS']['AXIS2'][0]
    yaxismax = fhere['AXIS']['AXIS2'][1]

    plt.imshow(fhere[dataset][:,:],
               aspect='auto',
               extent=[xaxismin, xaxismax, yaxismin, yaxismax])
    c=plt.colorbar(orientation='vertical')
    c.set_label(dataset)

    if(xlim != [-1,-1]):
        plt.xlim(xlim)
    if(ylim != [-1,-1]):
        plt.ylim(ylim)
    if(zlim != [-1,-1]):
        plt.clim(zlim)

    plt.show()


def phasespace_2d(rundir='',dataset='p1x1',species='electrons',time=0,
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
    enc = "utf-8"

    plt.figure(figsize=(6, 3.2))
    plt.title(dataset+' phasespace at t = '+str(fhere.attrs['TIME'][0]))
    plt.xlabel('$'+str(fhere['AXIS/AXIS1'].attrs['LONG_NAME'][0],enc)+' ['+ \
        str(fhere['AXIS/AXIS1'].attrs['UNITS'][0],enc)+']$')
    plt.ylabel('$'+str(fhere['AXIS/AXIS2'].attrs['LONG_NAME'][0],enc)+' ['+ \
        str(fhere['AXIS/AXIS2'].attrs['UNITS'][0],enc)+']$')

    xaxismin = fhere['AXIS/AXIS1'][0]
    xaxismax = fhere['AXIS/AXIS1'][1]
    yaxismin = fhere['AXIS/AXIS2'][0]
    yaxismax = fhere['AXIS/AXIS2'][1]

    if(zlim != [-1,-1]):
        norm = LogNorm(zlim[0],zlim[1])
        offset = 0
    else:
        offset = 1e-12
        norm = LogNorm(offset,np.max(np.abs(fhere[dataset][:,:])))

    plt.imshow(np.abs(fhere[dataset][:,:])+offset,
               aspect='auto',
               extent=[xaxismin, xaxismax, yaxismin, yaxismax],
               norm=norm)
    c=plt.colorbar(orientation='vertical')
    c.set_label(str(fhere[dataset].attrs['UNITS'][0],enc))


    if(xlim != [-1,-1]):
        plt.xlim(xlim)
    if(ylim != [-1,-1]):
        plt.ylim(ylim)

    plt.show()


def fieldinteract_2d(rundir='',dataset='e1',
    xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[],const_clim=True):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'FLD', dataset)
    files = sorted(os.listdir(odir))
    enc = "utf-8"

    data = []
    axes_lim = []
    norm = []
    for i in range(len(files)):
        fhere = h5py.File(os.path.join(odir,files[i]), 'r')
        data.append([fhere[dataset][:,:],fhere.attrs['TIME'][0],fhere.attrs['DT']])
        axes_lim.append(np.array([fhere['AXIS/AXIS1'][:],fhere['AXIS/AXIS2'][:]]).flatten())
        xlabel = '$'+str(fhere['AXIS/AXIS1'].attrs['LONG_NAME'][0],enc)+' ['+ \
                 str(fhere['AXIS/AXIS1'].attrs['UNITS'][0],enc)+']$'
        ylabel = '$'+str(fhere['AXIS/AXIS2'].attrs['LONG_NAME'][0],enc)+' ['+ \
                 str(fhere['AXIS/AXIS2'].attrs['UNITS'][0],enc)+']$'
        clabel = '$'+str(fhere[dataset].attrs['LONG_NAME'][0],enc)+' ['+ \
                 str(fhere[dataset].attrs['UNITS'][0],enc)+']$'

    if(zlim == [-1,-1]):
        if(const_clim):
            min_val = np.min([ np.min(x[0]) for x in data])
            max_val = np.max([ np.max(x[0]) for x in data])

    def fu(n):
        plt.figure(figsize=(8, 4))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.imshow(data[n][0],
               extent=axes_lim[n],
               aspect='auto',)
        plt.title('time = '+str(data[n][1]))
        c=plt.colorbar()
        c.set_label(clabel)
        if(xlim != [-1,-1]):
            plt.xlim(xlim)
        if(ylim != [-1,-1]):
            plt.ylim(ylim)
        if(zlim != [-1,-1]):
            plt.clim(zlim)
        else:
            if(const_clim):
                plt.clim([min_val,max_val])
        return plt

    interact(fu,n=(0,len(data)-1))


def phaseinteract_2d(rundir='',dataset='p1x1',species='electrons',
    xlim=[-1,-1],ylim=[-1,-1],zlim=[-1,-1],
    plotdata=[],const_clim=True):

    workdir = os.getcwd()
    workdir = os.path.join(workdir, rundir)

    odir = os.path.join(workdir, 'MS', 'PHA', dataset, species)
    files = sorted(os.listdir(odir))
    enc = "utf-8"
    if(zlim != [-1,-1]):
        offset = 0
    else:
        offset = 1e-12

    data = []
    axes_lim = []
    norm = []
    for i in range(len(files)):
        fhere = h5py.File(os.path.join(odir,files[i]), 'r')
        data.append([np.abs(fhere[dataset][:,:])+offset,fhere.attrs['TIME'][0],fhere.attrs['DT']])
        axes_lim.append(np.array([fhere['AXIS/AXIS1'][:],fhere['AXIS/AXIS2'][:]]).flatten())
        xlabel = '$'+str(fhere['AXIS/AXIS1'].attrs['LONG_NAME'][0],enc)+' ['+ \
                 str(fhere['AXIS/AXIS1'].attrs['UNITS'][0],enc)+']$'
        ylabel = '$'+str(fhere['AXIS/AXIS2'].attrs['LONG_NAME'][0],enc)+' ['+ \
                 str(fhere['AXIS/AXIS2'].attrs['UNITS'][0],enc)+']$'
        clabel = '$'+str(fhere[dataset].attrs['LONG_NAME'][0],enc)+' ['+ \
                 str(fhere[dataset].attrs['UNITS'][0],enc)+']$'
        if(zlim != [-1,-1]):
            norm.append(LogNorm(zlim[0],zlim[1]))
        else:
            norm.append(LogNorm(offset,np.max(data[-1][0])))

    if(zlim == [-1,-1]):
        if(const_clim):
            norm=[]
            max_val = np.max([ np.max(x[0]) for x in data])
            for i in range(len(files)):
                norm.append(LogNorm(offset,max_val))

    def fu(n):
        plt.figure(figsize=(8, 4))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.imshow(data[n][0],
               extent=axes_lim[n],
               aspect='auto',
               norm=norm[n])
        plt.title('time = '+str(data[n][1]))
        c=plt.colorbar()
        c.set_label(clabel)
        if(xlim != [-1,-1]):
            plt.xlim(xlim)
        if(ylim != [-1,-1]):
            plt.ylim(ylim)
        return plt

    interact(fu,n=(0,len(data)-1))



def tajima(rundir):
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
        phase_space_dir=working_dir+'/MS/PHA/p1x1/electrons/'
        p2x1_dir=working_dir+'/MS/PHA/p2x1/electrons/'

        efield_prefix='e1-'
        laser_prefix='e2-'
        phase_prefix='p1x1-electrons-'
        p2x1_prefix='p2x1-electrons-'
        eden_prefix='charge-electrons-'
        plt.figure(figsize=(12,16))

        filename1=phase_space_dir+phase_prefix+repr(file_no).zfill(6)+'.h5'
        filename2=eden_dir+eden_prefix+repr(file_no).zfill(6)+'.h5'
        filename3=efield_dir+efield_prefix+repr(file_no).zfill(6)+'.h5'
        filename4=laser_dir+laser_prefix+repr(file_no).zfill(6)+'.h5'
        filename5=p2x1_dir+p2x1_prefix+repr(file_no).zfill(6)+'.h5'

        #print(filename1)
        #print(filename2)

        phase_space=np.abs(osh5io.read_h5(filename1))
        # print(repr(phase_space))
        eden=osh5io.read_h5(filename2)
        ex = osh5io.read_h5(filename3)
        ey = osh5io.read_h5(filename4)
        p2x1=np.abs(osh5io.read_h5(filename5))

        phase_plot=plt.subplot(325)
        #print(repr(phase_space.axes[0].min))
        #print(repr(phase_space.axes[1].min))
        title=phase_space.data_attrs['LONG_NAME']
        time=phase_space.run_attrs['TIME'][0]
        ext_stuff=[phase_space.axes[1].min,phase_space.axes[1].max,phase_space.axes[0].min,phase_space.axes[0].max]
        data_max=max(np.abs(np.amax(phase_space)),100)
        print(repr(data_max))
        phase_contour=plt.contourf(np.abs(phase_space+0.000000001),
                    levels=[0.00001*data_max,0.0001*data_max,0.001*data_max,0.01*data_max,0.05*data_max,0.1*data_max,0.2*data_max,0.5*data_max],
                    extent=ext_stuff,cmap='Spectral',vmin=1e-5*data_max,vmax=1.5*data_max,
                    norm=colors.LogNorm(vmin=0.00001*data_max,vmax=1.5*data_max))
        phase_plot.set_title('P2X1 Phase Space' +' , t='+repr(time)+' $\omega_{pe}^{-1}$')
        phase_plot.set_xlabel('Position [$\Delta x$]')
        phase_plot.set_ylabel('Velocity [$\omega_{pe} \Delta x$]')
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        # plt.contour(np.abs(phase_space+0.000001),levels=[0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1],extent=ext_stuff,colors='black',linestyles='dashed')
        plt.colorbar(phase_contour)
        
        
        den_plot = plt.subplot(321)
        osh5vis.osplot(eden,title='Electron Density',ylim=[-2,0])
        
        ex_plot = plt.subplot(322)
        
        osh5vis.osplot(ex,title='Wake electric field')
        
        ey_plot = plt.subplot(323)
        
        osh5vis.osplot(ey,title='Laser electric field')
        
        ey_plot_k = plt.subplot(324)
        
        
        osh5vis.osplot(np.abs(osh5utils.fft(ey)), xlim=[0, 20], ylim=[0, 300],linestyle='-')
        
        # plt.plot(ex[0,:])
        # plt.ylim([-2,2])
        # ex_plot.set_xlabel('Position [$\Delta x$]')
        # ex_plot.set_ylabel('Electric Field')
        # plt.tight_layout()
        # plt.show()
        
        p2x1_plot=plt.subplot(326)
        #print(repr(phase_space.axes[0].min))
        #print(repr(phase_space.axes[1].min))
        title=p2x1.data_attrs['LONG_NAME']
        time=p2x1.run_attrs['TIME'][0]
        ext_stuff=[p2x1.axes[1].min,p2x1.axes[1].max,p2x1.axes[0].min,p2x1.axes[0].max]
        p2x1_contour=plt.contourf(np.abs(p2x1+0.000000001),levels=[0.00001,0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1,10,100,500],extent=ext_stuff,cmap='Spectral',vmin=1e-5,vmax=3000,
                    norm=colors.LogNorm(vmin=0.0001,vmax=3000))
        p2x1_plot.set_title('Phase Space' +' , t='+repr(time)+' $\omega_{pe}^{-1}$')
        p2x1_plot.set_xlabel('Position [$\Delta x$]')
        p2x1_plot.set_ylabel('Velocity [$\omega_{pe} \Delta x$]')
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        # plt.contour(np.abs(phase_space+0.000001),levels=[0.0001,0.001,0.01,0.05,0.1,0.2,0.5,1],extent=ext_stuff,colors='black',linestyles='dashed')
        plt.colorbar(p2x1_contour)
        
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

    interact(something,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0, continous_update=False))
    #something(rundir=rundir,file_no=20)
    
    



def phasespace_movie(rundir):
#2345
    import os


    def something(rundir,file_no):

        my_path=os.getcwd()
        #print(my_path)
        working_dir=my_path+'/'+rundir
        #print(working_dir)
        efield_dir=working_dir+'/DIAG/Ex/'
        phase_space_dir=working_dir+'/DIAG/Vx_x/'
        ex_prefix='Ex-0_'
        phase_prefix='vx_x_'
        plt.figure(figsize=(12,6))

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
        phase_contour=plt.contourf(phase_space,levels=[0.1,1,2,3,5,10,100,1000,100000],extent=ext_stuff,cmap='Spectral',vmin=1e-1,vmax=100000,
                    norm=colors.LogNorm(vmin=0.1,vmax=100000))
        phase_plot.set_title('Phase Space' +' , t='+repr(time)+' $\omega_{pe}^{-1}$')
        phase_plot.set_xlabel('Position [$\Delta x$]')
        phase_plot.set_ylabel('Velocity [$\omega_{pe} \Delta x$]')
        #plt.colorbar()
        #osh5vis.oscontour(phase_space,levels=[10**-5,10**-3,10**-1,1,10,100],colors='black',linestyles='dashed',vmin=1e-5,vmax=1000)
        plt.contour(phase_space,levels=[0.1,1,2,3,5,10,100,1000,100000],extent=ext_stuff,colors='black',linestyles='dashed')
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
    print(files[1][start:end])
    file_interval=int(files[1][start:end])
    file_max=(len(files)-1)*file_interval

    interact(something,rundir=fixed(rundir),file_no=widgets.IntSlider(min=0,max=file_max,step=file_interval,value=0))
    #something(rundir=rundir,file_no=20)

