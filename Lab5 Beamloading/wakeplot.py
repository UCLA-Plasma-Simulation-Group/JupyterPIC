import os
import sys
import shutil
import subprocess
from IPython.display import display
import h5py
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider, SelectionSlider
import ipywidgets as widgets


def z1_interact(box_z,indz,nb_d,z_d,sigr_d,sigz_d,nb_w,sigr_w,z1_w,z2_w,use_fixed_length,use_theory):
    
    if nb_w>0 and not(z1_w>np.pi/2.0 and z1_w<np.pi):
        sys.exit('Error: trying to load beam in the wrong region')
       
    if nb_w<0 and not(z1_w>1.5*np.pi and z1_w<2.0*np.pi):
        sys.exit('Error: trying to load beam in the wrong region')
    
    R0th=float(sigr_d**2/2.0*np.exp(sigr_d**2/2.0)*mp.gammainc(0,sigr_d**2/2.0))
    R0thw=float(sigr_w**2/2.0*np.exp(sigr_w**2/2.0)*mp.gammainc(0,sigr_w**2/2.0))
    
    olength=z2_w-z1_w
    nb_w0=nb_w
            
    ximin=box_z[0]
    ximax=box_z[1]
    
    shape_alpha=0.9
    
    def plot_wakes(z1_w):
          
            nb_wlocal=0.0
            z2_wlocal=0.0
            if use_theory==True:
                z2_wlocal=z1_w-np.tan(z1_w)
                nb_wlocal=nb_d*np.sqrt(2.0*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)*np.sin(z1_w)*R0th/R0thw
            else: 
                nb_wlocal=nb_w
                if use_fixed_length==True:
                    z2_wlocal=z1_w+olength
                else:
                    z2_wlocal=z2_w
            
            figure_title = '$E_z$'
            xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
            ylabel_left ='$eE_z/mc\omega_p$'
            ylabel_right= '$n_b/n_p$'
            datamin = -1.0
            datamax = 1.0
            xi=np.linspace(ximin,ximax,2**indz)

            fig, ax1 = plt.subplots(figsize=(8,5))
            #plt.axis([ xi.min(), xi.max(),datamin, datamax])
            plt.title(figure_title)
            ax1.set_xlabel(xlabel_bottom)
            ax1.set_ylabel(ylabel_left, color='k')
            ax1.tick_params('y', colors='k')
            ax1.set_xlim([xi.min(),xi.max()])

            ax2=ax1.twinx()
            ax2.set_ylabel(ylabel_right, color='b')
            ax2.tick_params('y', colors='b')
            nbmax=np.max(np.array([nb_d,nb_wlocal]))
            ax2.set_ylim([-2.0*nbmax,2.0*nbmax])

            temp=xi[xi<z_d+3.0*sigr_d]
            xid=temp[temp>z_d-3.0*sigr_d]
            d_profile=nb_d*np.exp(-(xid-z_d)**2/(2.0*sigz_d**2))
            ax2.fill_between(xid,0,d_profile,alpha=shape_alpha)

            temp=xi[xi>z1_w+z_d]
            xiw=temp[temp<z2_wlocal+z_d]
            w_profile=nb_wlocal*(z2_wlocal+z_d-xiw)/(z2_wlocal-z1_w)
            ax2.fill_between(xiw,0,w_profile,alpha=shape_alpha)

            def Zpth(kpxi):
                return nb_d*np.sqrt(2.*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)*np.cos(kpxi)
            def Ezth(kpxi):
                return Zpth(kpxi)*R0th

            def Zpthw(kpxi):
                if kpxi<z2_wlocal:
                    return nb_wlocal/(z2_wlocal-z1_w)*( (z2_wlocal-z1_w)*np.sin(kpxi-z1_w)+np.cos(kpxi-z1_w)-1.0)
                else:
                    return nb_wlocal/(z2_wlocal-z1_w)*( (z2_wlocal-z1_w)*np.sin(kpxi-z1_w)+np.cos(kpxi-z1_w)-np.cos(kpxi-z2_wlocal))
            def Ezthw(kpxi):
                return [Zpthw(i)*R0thw for i in kpxi]

            ezmaxd=np.max(np.abs(Ezth(xi[xi>z_d]-z_d)))
            ezmaxw=np.max(np.abs(Ezthw(xi[xi>z1_w+z_d]-z_d)))
            ezmaxt=np.max(np.abs(Ezthw(xi[xi>z1_w+z_d]-z_d)+Ezth(xi[xi>z1_w+z_d]-z_d)))
            ezmax=np.max(np.array([float(ezmaxd),float(ezmaxw),float(ezmaxt)]))
            ax1.set_ylim([-1.1*ezmax,1.1*ezmax])

            Eztot=np.copy(Ezth(xi-z_d))
            Ezw=np.copy(Ezthw(xi-z_d))
            Eztot[xi>z1_w+z_d]=np.add(Eztot[xi>z1_w+z_d],Ezw[xi>z1_w+z_d])
            ax1.plot(xi[xi>z_d],Ezth(xi[xi>z_d]-z_d),'C0',label='wake from driver')
            ax1.plot(xi[xi>z1_w+z_d],Ezthw(xi[xi>z1_w+z_d]-z_d),'k',linewidth=1.5)
            ax1.plot(xi[xi>z1_w+z_d],Ezthw(xi[xi>z1_w+z_d]-z_d),'C1',linewidth=1.4,label='wake from witness beam')
            ax1.plot(xi[xi>z_d],Eztot[xi>z_d],'r--',label='total wake')
            #ax1.plot(xi[xi>z1_w+z_d],Ezthw(xi[xi>z1_w+z_d]-z_d)+Ezth(xi[xi>z1_w+z_d]-z_d),'r',label='total wake')

            #ax1.plot(xi,Ezth(xi-z_d),label='wake from driver')
            #ax1.plot(xi,Ezthw(xi-z_d),label='wake from witness beam')
            #ax1.plot(xi,Ezthw(xi-z_d)+Ezth(xi-z_d),'r--',label='total wake')
            ax1.legend(loc=3)
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.patch.set_visible(False)
            
            fig.tight_layout()
            
            if use_theory==True:
                print('Tail location = ','%.3f' % z2_wlocal)
                print('Beam density = ','%.3e' % nb_wlocal)
            
    def plot_wakes_nb(nb_w):
        
            if nb_w0<0:
                nb_w=-nb_w
                
            #if use_theory==True:
            #    z1_w=np.pi+np.arcsin(R0thw/R0th*nb_w/(nb_d*np.sqrt(2.0*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)))
            #    z2_w=z1_w+np.tan(z1_w)
                
            z1_wlocal=0.0
            z2_wlocal=0.0
            if use_theory==True:
                z1_wlocal=np.pi-np.arcsin(R0thw/R0th*nb_w/(nb_d*np.sqrt(2.0*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)))
                z2_wlocal=z1_wlocal-np.tan(z1_wlocal)
                if nb_w0<0:
                    z1_wlocal=2.0*np.pi-np.arcsin(-R0thw/R0th*nb_w/(nb_d*np.sqrt(2.0*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)))
                    z2_wlocal=z1_wlocal-np.tan(z1_wlocal)
            else: 
                z1_wlocal=z1_w
                z2_wlocal=z2_w
        

            figure_title = '$E_z$'
            xlabel_bottom = r'$\xi = ct-z\;[c/\omega_p]$'  
            ylabel_left ='$eE_z/mc\omega_p$'
            ylabel_right= '$n_b/n_p$'
            datamin = -1.0
            datamax = 1.0
            xi=np.linspace(ximin,ximax,2**indz)

            fig, ax1 = plt.subplots(figsize=(8,5))
            #plt.axis([ xi.min(), xi.max(),datamin, datamax])
            plt.title(figure_title)
            ax1.set_xlabel(xlabel_bottom)
            ax1.set_ylabel(ylabel_left, color='k')
            ax1.tick_params('y', colors='k')
            ax1.set_xlim([xi.min(),xi.max()])

            def Zpth(kpxi):
                return nb_d*np.sqrt(2.*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)*np.cos(kpxi)
            def Ezth(kpxi):
                return Zpth(kpxi)*R0th

            def Zpthw(kpxi):
                if kpxi<z2_wlocal:
                    return nb_w/(z2_wlocal-z1_wlocal)*( (z2_wlocal-z1_wlocal)*np.sin(kpxi-z1_wlocal)+np.cos(kpxi-z1_wlocal)-1.0)
                else:
                    return nb_w/(z2_wlocal-z1_wlocal)*( (z2_wlocal-z1_wlocal)*np.sin(kpxi-z1_wlocal)+np.cos(kpxi-z1_wlocal)-np.cos(kpxi-z2_wlocal))
            def Ezthw(kpxi):
                return [Zpthw(i)*R0thw for i in kpxi]

            ezmaxd=np.max(np.abs(Ezth(xi[xi>z_d]-z_d)))
            ezmaxw=np.max(np.abs(Ezthw(xi[xi>z1_wlocal+z_d]-z_d)))
            ezmaxt=np.max(np.abs(Ezthw(xi[xi>z1_wlocal+z_d]-z_d)+Ezth(xi[xi>z1_wlocal+z_d]-z_d)))
            ezmax=np.max(np.array([float(ezmaxd),float(ezmaxw),float(ezmaxt)]))
            ax1.set_ylim([-1.1*ezmax,1.1*ezmax])

            Eztot=np.copy(Ezth(xi-z_d))
            Ezw=np.copy(Ezthw(xi-z_d))
            Eztot[xi>z1_wlocal+z_d]=np.add(Eztot[xi>z1_wlocal+z_d],Ezw[xi>z1_wlocal+z_d])
            ax1.plot(xi[xi>z_d],Ezth(xi[xi>z_d]-z_d),'C0',label='wake from driver')
            ax1.plot(xi[xi>z1_wlocal+z_d],Ezthw(xi[xi>z1_wlocal+z_d]-z_d),'k',linewidth=1.5)
            ax1.plot(xi[xi>z1_wlocal+z_d],Ezthw(xi[xi>z1_wlocal+z_d]-z_d),'C1',linewidth=1.4,label='wake from witness beam')
            ax1.plot(xi[xi>z_d],Eztot[xi>z_d],'r--',label='total wake')
            #ax1.plot(xi[xi>z1_w+z_d],Ezthw(xi[xi>z1_w+z_d]-z_d)+Ezth(xi[xi>z1_w+z_d]-z_d),'r',label='total wake')

            #ax1.plot(xi,Ezth(xi-z_d),label='wake from driver')
            #ax1.plot(xi,Ezthw(xi-z_d),label='wake from witness beam')
            #ax1.plot(xi,Ezthw(xi-z_d)+Ezth(xi-z_d),'r--',label='total wake')
            ax1.legend(loc=3)

            ax2=ax1.twinx()
            ax2.set_ylabel(ylabel_right, color='b')
            ax2.tick_params('y', colors='b')
            nbmax=np.max(np.array([nb_d,nb_w]))
            ax2.set_ylim([-2.0*nbmax,2.0*nbmax])

            temp=xi[xi<z_d+3.0*sigr_d]
            xid=temp[temp>z_d-3.0*sigr_d]
            d_profile=nb_d*np.exp(-(xid-z_d)**2/(2.0*sigz_d**2))
            ax2.fill_between(xid,0,d_profile,alpha=shape_alpha)

            temp=xi[xi>z1_wlocal+z_d]
            xiw=temp[temp<z2_wlocal+z_d]
            w_profile=nb_w*(z2_wlocal+z_d-xiw)/(z2_wlocal-z1_wlocal)
            ax2.fill_between(xiw,0,w_profile,alpha=shape_alpha)
            
            ax1.set_zorder(ax2.get_zorder()+1)
            ax1.patch.set_visible(False)

            fig.tight_layout()           
        
            if use_theory==True:
                print('Head location = ','%.3f' % z1_wlocal)
                print('Tail location = ','%.3f' % z2_wlocal)
    
    z1_wmin=1.6
    z1_wmax=3.1
    if nb_w<0:
        z1_wmin+=np.pi
        z1_wmax+=np.pi
    i1=interact(plot_wakes,z1_w=FloatSlider(min=z1_wmin,max=z1_wmax,step=0.05,value=z1_w,continuous_update=False))
    
    nb_wmax=nb_d*np.sqrt(2.0*np.pi)*sigz_d*np.exp(-sigz_d**2/2.)*R0th/R0thw*0.99
    
    #if nb_w0>0:
    #    i2=interact(plot_wakes_nb,nb_w=FloatSlider(min=nb_wmax/10.0,max=nb_wmax,step=nb_wmax/10.0,value=nb_w0,continuous_update=False))
    #else:
    #    i2=interact(plot_wakes_nb,nb_w=FloatSlider(min=nb_wmax/10.0,max=nb_wmax,step=nb_wmax/10.0,value=-nb_w0,continuous_update=False))
        
    values=np.arange(nb_wmax/10.0,1.01*nb_wmax,nb_wmax/10.0)
    i2=interact(plot_wakes_nb,nb_w=SelectionSlider(options=[("%.3e"%i,i) for i in values],continuous_update=False))
