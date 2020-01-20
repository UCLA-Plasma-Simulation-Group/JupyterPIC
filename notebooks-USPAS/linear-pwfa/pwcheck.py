import h5py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, FloatSlider, HBox, VBox, interactive_output
   
def makeplot(rundir):
    def plot_pw(xi_position):
        filename=rundir+'/Fields/Ez_slice0001/ezslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        zCellsTotal=data.shape[0]
        xCellsTotal=data.shape[1]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        zCellsPerUnitLength=zCellsTotal/(yaxis[1]-yaxis[0])
        xCellsPerUnitLength=xCellsTotal/(xaxis[1]-xaxis[0])
        deltax=1.0/xCellsPerUnitLength
        deltaz=1.0/zCellsPerUnitLength
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[1]/2)) 
        x=np.linspace(xaxis[0],xaxis[1],int(data.shape[1])) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[0])         
        figure_title = r'$\partial_r E_z,\;\partial_\xi F_\perp\;[e/mc^2]$'
        xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
        xlabel_top = '$eE_z/mc\omega_p$'
        ylabel_left ='$x\;[c/\omega_p]$'
        ylabel_right ='$eE_z/mc\omega_p$'
        datamin = -1.0
        datamax = 1.0
        colormap = 'bwr'
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        da = data
        l_max = 1
        l_min = -1
        fig, ax1 = plt.subplots(figsize=(8,5))
        plt.axis([ xi.min(), xi.max(),x.min(), x.max()])
        plt.title(figure_title)
        
        deriv=da.copy()
        deriv[1:-1,:]=(da[2:,:]-da[0:-2,:])/(2.0*deltax)
        deriv[0,:]=(da[1,:]-da[0,:])/deltax
        deriv[-1,:]=(da[-1,:]-da[-2,:])/deltax
        
        cs1 = plt.pcolormesh(xi,x,deriv,cmap=colormap)
        plt.colorbar(cs1, pad = 0.15)
        ax1.set_xlabel(xlabel_bottom)
        ax1.set_ylabel(ylabel_left, color='k')
        ax1.tick_params('y', colors='k')
        ax2 = ax1.twiny()
        lineout_index = int (xi_position * zCellsPerUnitLength)
        lineout = da[:,lineout_index]
       
        ax2.plot(deriv[:,lineout_index],x,'b',label='Derivative')
        ax1.plot(xi_position*np.ones(da.shape[0]),x, 'k--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('x', colors='r')

        filename=rundir+'/Fields/Ex_slice0001/exslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        
        ax2.plot(deriv[:,lineout_index],x,'b',label='Derivative')
        ax1.plot(xi_position*np.ones(da.shape[0]),x, 'k--')
        ax2.set_ylabel(xlabel_top, color='r')
        ax2.tick_params('x', colors='r')

        filename=rundir+'/Fields/Ex_slice0001/exslicexz_00000001.h5'
        f=h5py.File(filename,'r')
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=dataset[...]
        xaxis=f['/AXIS/AXIS1'][...]
        yaxis=f['/AXIS/AXIS2'][...]
        x=np.linspace(xaxis[0]/2,xaxis[1]/2,int(data.shape[1]/2)) 
        x=np.linspace(xaxis[0],xaxis[1],int(data.shape[1])) 
        xi=np.linspace(yaxis[0],yaxis[1],data.shape[0])         
        filename=rundir+'/Fields/By_slice0001/byslicexz_00000001.h5'
        f=h5py.File(filename)
        names=list(f.keys())
        dataname='/'+names[1]
        dataset=f[dataname]
        data=-data+dataset[...]
        data = data.transpose()
        da = data[int(xCellsTotal/4):int(3*xCellsTotal/4),:] 
        da = data
        l_max = 1.0
        l_min = -1.0

        deriv=da.copy()
        deriv[:,1:-1]=(da[:,2:]-da[:,0:-2])/(2.0*deltaz)
        deriv[:,0]=(da[:,1]-da[:,0])/deltaz
        deriv[:,-1]=(da[:,-1]-da[:,-2])/deltaz

        ax2.plot(deriv[:,lineout_index],x,'r--',label='deriv of foc')

        fig.tight_layout()
    
    i1=interact(plot_pw,xi_position=FloatSlider(min=0,max=10,step=0.05,value=5.75,continuous_update=False))
    return
   