# in the output data folder. For testing purpose
import os
import shutil
import subprocess
from IPython.display import display
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual
from h5_utilities import *
from analysis import *
from scipy.optimize import fsolve

def makeplot():
    def plot1_qp(x_position):
        filename='Species0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        figure_title = 'Plasma Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$Plasma Density\;[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$Plasma Density\;[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'viridis'
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 2.0
        l_min = -10.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * 21.333333333333333)+1
        lineout = da[lineout_index,:]
        ax2.plot(y, lineout, 'r')
        ax2.set_ylim([l_min,l_max])
        ax1.plot(y, x_position*np.ones(256), 'w--') # Add a white dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot2_qp(xi_position):
        filename='Species0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1]) 
        figure_title = 'Plasma Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$Plasma Density\;[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$Plasma Density\;[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'viridis'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 2.0
        l_min = -4.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * 25.6) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r')
        ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(128),x, 'w--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot1_qb(x_position):
        filename='Beam0002/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        filename='Beam0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=data+dataset[...]
        figure_title = 'Beam Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Beam Density $[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Beam Density $[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'afmhot'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 0.0
        l_min = -30.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * 21.333333333333333)+1 
        lineout = da[lineout_index,:]
        ax2.plot(y, lineout, 'r')
        ax2.set_ylim([l_min,l_max])
        ax1.plot(y, x_position*np.ones(256), 'b--') # Add a white dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot2_qb(xi_position):
        filename='Beam0001/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        filename='Beam0002/Charge_slice_0001/charge_slice_xz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=data+dataset[...]
        figure_title = 'Beam Density'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Beam Density $[n_p]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Beam Density $[n_p]$'
        datamin = -10.0
        datamax = 0.0
        colormap = 'afmhot'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 0.0
        l_min = -30.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * 25.6) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r')
        ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(128),x, 'b--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot1_ez(x_position):
        filename='Fields/Ez_slice0001/ezslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        figure_title = '$E_z$'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$eE_z/mc\omega_p$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$eE_z/mc\omega_p$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'bwr'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 1.0
        l_min = -1.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * 21.333333333333333) 
        lineout = da[lineout_index,:]
        ax2.plot(y, lineout, 'r')
        ax2.set_ylim([l_min,l_max])
        ax1.plot(y, x_position*np.ones(256), 'w--') # Add a white dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot2_ez(xi_position):
        filename='Fields/Ez_slice0001/ezslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        figure_title = '$E_z$'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$eE_z/mc\omega_p$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$eE_z/mc\omega_p$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'bwr'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 0.5
        l_min = -0.5
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * 25.6) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r')
        ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(128),x, 'w--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot1_fo(x_position):
        filename='Fields/Ex_slice0001/exslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        filename='Fields/By_slice0001/byslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=data-dataset[...]
        figure_title = 'Focusing Field'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Focusing Field $[mc\omega_p/e]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Focusing Field $[mc\omega_p/e]$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'jet'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 1.0
        l_min = -1.0
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        middle_index = int(da.shape[0]/2)
        lineout_index = int (middle_index + x_position * 21.333333333333333)+1 
        lineout = da[lineout_index,:]
        ax2.plot(y, lineout, 'r')
        ax2.set_ylim([l_min,l_max])
        ax1.plot(y, x_position*np.ones(256), 'b--') # Add a white dashed line at the lineout position
        ax2.set_ylabel(ylabel_right, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return

    def plot2_fo(xi_position):
        filename='Fields/Ex_slice0001/exslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[0]/2)) 
        y=np.linspace(yaxis[0],yaxis[1],data.shape[1])         
        filename='Fields/By_slice0001/byslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=data-dataset[...]
        figure_title = 'Focusing Field'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = 'Focusing Field $[mc\omega_p/e]$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='Focusing Field $[mc\omega_p/e]$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'jet'
        data = data.transpose()
        da = data[64:192,:] 
        l_max = 0.6
        l_min = -0.6
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ y.min(), y.max(),x.min(), x.max()])
        plt.title(figure_title)
        cs1 = plt.pcolormesh(y,x,da,vmin=datamin,vmax=datamax,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * 25.6) 
        lineout = da[:,lineout_index]
        ax2.plot(lineout, x, 'r')
        ax2.set_xlim([l_min,l_max])
        ax1.plot(xi_position*np.ones(128),x, 'b--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        return        

    interact(plot1_qp,x_position=(-3,3,0.05))
    interact(plot2_qp,xi_position=(0,10,0.05))
    interact(plot1_qb,x_position=(-3,3,0.05))
    interact(plot2_qb,xi_position=(0,10,0.05))
    interact(plot1_ez,x_position=(-3,3,0.05))
    interact(plot2_ez,xi_position=(0,10,0.05))
    interact(plot1_fo,x_position=(-3,3,0.05))
    interact(plot2_fo,xi_position=(0,10,0.05))
    return
