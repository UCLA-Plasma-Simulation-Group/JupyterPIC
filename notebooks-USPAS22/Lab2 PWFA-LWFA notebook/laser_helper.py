import sys
b= sys.path
sys.path=['/home/jovyan/analysis'] + b
import osiris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from osh5vis import new_fig, osplot
from osh5io import read_h5
from osh5utils import fft
import glob
from ipywidgets import interact_manual,fixed,Layout,interact, FloatSlider
from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual, FloatSlider, HBox, VBox, interactive_output
import ipywidgets as widgets
interact_calc=interact_manual.options(manual_name="Make New Input and Run")
import os
import h5py as h5
from osiris import tajima


def newifile(iname='os-stdin', oname='output.txt', 
             a0=4.0, omega0=36.0,w0 =4.0 , t_flat=0, t_rise=5.6, t_fall=5.6,
             ndump=2,  tmax=200.0,nodex =1, nodez=1 ):

    with open(iname) as osdata:
        data = osdata.readlines()
    nproc = nodex*nodez
    for i in range(len(data)):
        if 'node_number(1:2) =' in data[i]:
      
             data[i] = '    node_number(1:2) = '+str(nodez)+','+ str(nodex)+',\n'
        if 'ndump =' in data[i]:
            data[i] = '    ndump = '+str(ndump)+',\n'
        if 'tmax =' in data[i]:
            data[i] = '    tmax ='+str(tmax)+'\n'    
        if 'a0 =' in data[i] and 'omega0' not in data[i]:
            data[i] = '    a0= '+str(a0)+',\n'
        if 'omega0 =' in data[i]:
            data[i] = '    omega0 = '+str(omega0)+',\n'
        if 'lon_flat =' in data[i]:
            data[i] = '    lon_flat = '+str(t_flat)+',\n'
        if 'lon_rise =' in data[i]:
            data[i] = '    lon_rise = '+str(t_rise)+',\n'
        if 'lon_fall =' in data[i]:
            data[i] = '    lon_fall = '+str(t_fall)+',\n'
        if 'per_w0 =' in data[i]:
           
            if str(oname[:-3]) =="linear":
                data[i] = '    per_w0 = '+str(w0)+',\n'
            if str(oname[:-3])=='nonlinear':
                data[i] = '    per_w0 = '+str(2*np.sqrt(a0))+',\n'
                print('The matched spot size W0 is', str(2*np.sqrt(a0)))
        # This line is for moving window only
      
       

    with open(oname,'w') as f:
        for line in data:
            f.write(line)
    
    print('New file '+oname+' is written.')
    dirname = str(oname[:-3])
    print(dirname)
    print('Running OSIRIS in directory '+dirname+'...')
    osiris.runosiris_2d(rundir=dirname,inputfile=oname, print_out='yes',combine='no',np=nproc)
    print('The spot size W0 is adjusted to the matched spot size', str(2*np.sqrt(a0)))
    
    print('Done')
 
def moving_widget(regime):
    
    style = {'description_width': '350px'}
    layout = Layout(width='55%')
    
    if regime =='linear':
        a0 = 0.5
        w0 = 6
        fname = 'linear.2d'
    if regime == 'nonlinear':
        a0 = 4.0
        w0 = 2*np.sqrt(a0)
        fname = 'nonlinear.2d'

    a = widgets.Text(value='os-stdin', description='Template Input File:',style=style,layout=layout)
    b = widgets.Text(value=fname, description='New Output File:',style=style,layout=layout)
    #velocity
   
    # a0
    d = widgets.FloatText(value=a0,description='a0:',style=style,layout=layout)
    # frequency 
    e = widgets.BoundedFloatText(value=36.2925, min=0, max=50, description='omega0:',style=style,layout=layout)
    # time 
    f = widgets.BoundedFloatText(value=0, min=0, max=100, description='t_flat:',style=style,layout=layout)
    g = widgets.BoundedFloatText(value=5.64152, min=0, max=100, description='t_rise:',style=style,layout=layout)
    h = widgets.BoundedFloatText(value=5.64152, min=0, max=100, description='t_fall:',style=style,layout=layout)
    
    w0 = widgets.BoundedFloatText(value=w0, min=0.0, max=10.0, description='spot size W0:',style=style,layout=layout)
    #xmaxw = widgets.FloatText(value=24, description='xmax:', style=style, layout=layout)
    ndumpw = widgets.IntText(value=3,min = 3, max = 10, description='ndump:', style=style, layout=layout)
    #ppc = widgets.IntText(value=2, min = 1, max = 3, description='Particles per cell:', style=style, layout=layout)
    tmaxw = widgets.FloatText(value=40, description='tmax:', style=style, layout=layout)
    #nx_pw = widgets.IntText(value=2400, description="Number of cells:", style=style, layout=layout)
    
    nodex = widgets.IntText(value=2, min=0, max=2,description="Nodes in x:", style=style, layout=layout)
    nodez = widgets.IntText(value=2, min=0, max=2,description="Nodes in z:", style=style, layout=layout)

    
    im_moving = interact_calc(newifile, iname=a,oname=b,a0=d,omega0=e,w0=w0, t_flat=f, 
                  t_rise=g, t_fall=h, ndump=ndumpw,  tmax=tmaxw,nodex=nodex,nodez=nodez)
    im_moving.widget.manual_button.layout.width='250px'


def makeplot(file_id,path,field):
    
   

    def getdata(id):
        f = h5.File(path[id],"r")
        datasetNames = [n for n in f.keys()] #Two Datasets: AXIS and e2
        field = datasetNames[-1]
        Field_dat = f[field][:].astype(float)

        a1_bounds = f['AXIS']['AXIS1']
        a2_bounds = f['AXIS']['AXIS2']

        xi_dat = np.linspace(0,a1_bounds[1]-a1_bounds[0] ,len(Field_dat[0]))
        r_dat = np.linspace(a2_bounds[0],a2_bounds[1],len(Field_dat))
        return Field_dat, r_dat, xi_dat
    def den_plot(x_position):
        den,r,xi = getdata(-1)
        
        rindex = np.searchsorted(r, x_position, side="left")
        fig, ax = plt.subplots(figsize=(8,5))
        
        colors = ax.pcolormesh(xi,r,den,vmin=-2,vmax=0,cmap="Blues_r")
 
        cbar = fig.colorbar(colors,ax=ax,pad = 0.15)
        cbar.set_label('Density Field ($n_0$)')
        
        ax.hlines(r[rindex],xi[0],xi[-1],'k','--')
        ax2 = ax.twinx()
        ax2.plot(xi,den[rindex],'g',alpha = 0.7)
        ax2.tick_params('y', colors='g')
        ax.set_xlabel("$\\xi$ ($c/\omega_p$)")
        ax.set_ylabel('r ($c/\omega_p$)')
        ax.set_title('Density Field')


    def field_plot(field_id,x_position):
        id = int(field_id)
        field,r,xi = getdata(id)
        rindex = np.searchsorted(r, x_position, side="left")
        fig, ax = plt.subplots(figsize=(8,5))
    
        colors = ax.pcolormesh(xi,r,field,vmin=-field.max(),vmax=field.max(),cmap="RdBu_r")
    
        cbar = fig.colorbar(colors,ax=ax,pad = 0.15)
        if id == 0:
            name = 'Lonitudinal E'
        if id == 1:
            name = 'Transverse E'
        if id == 2:
            name = 'Transverse E'
        if id == 3:
            name = 'Longitudinal B'
        if id == 4:
            name = 'Transverse B'
        if id == 5:
            name = 'Transverse B'
        if id == 6:
            name = 'Psi'    
        cbar.set_label(name +' Field ($m_e c\omega_p / e$)')
        ax.hlines(r[rindex],xi[0],xi[-1],'k','--') 
        ax2 = ax.twinx()
        ax2.plot(xi,field[rindex],'g',alpha = 0.7)
        ax2.tick_params('y', colors='g')
       
        ax.set_xlabel("$\\xi$ ($c/\omega_p$)")
        ax.set_ylabel('r ($c/\omega_p$)')
        ax.set_title(name+' Field')
  
    if field == 'density':
        interact(den_plot,x_position=FloatSlider(min=-5,max=5,step=0.05,continuous_update=False))
    if field == 'field':
        interact(field_plot,field_id = FloatSlider(min=0,max=6,step=1,continuous_update=False),x_position=FloatSlider(min=-5,max=5,step=0.05,continuous_update=False))
    
