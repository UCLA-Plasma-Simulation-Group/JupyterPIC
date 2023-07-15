# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
#
# osh5utils_q3d.py
# quasi-3d utilities for pyVisOS
#
# Revision History
# Version 1:  First commit, made a subroutine that converts q3d data to full
#             3d data via mode summation
#
#
#
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************



import osh5io
import osh5def
import osh5vis
import osh5utils
import matplotlib.pyplot as plt
import osh5visipy


# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************




def filename_re(rundir,plasma_field,mode,fileno):
    plasma_field_path=rundir+'/MS/FLD/MODE-{mode:1d}-RE/{plasma_field:s}_cyl_m/{plasma_field:s}_cyl_m-{mode:1d}-re-{fileno:06d}.h5'.format(plasma_field=plasma_field,mode=mode,fileno=fileno)
    # print(plasma_field_path)
    return(plasma_field_path)

def filename_im(rundir,plasma_field,mode,fileno):
    plasma_field_path=rundir+'/MS/FLD/MODE-{mode:1d}-RE/{plasma_field:s}_cyl_m/{plasma_field:s}_cyl_m-{mode:1d}-re-{fileno:06d}.h5'.format(plasma_field=plasma_field,mode=mode,fileno=fileno)
    # print(plasma_field_path)
    return(plasma_field_path)

def filename_3d(rundir,plasma_field,fileno):
    filename=rundir+'/MS/FLD/{plasma_field:s}/{plasma_field:s}-{fileno:06d}.h5'.format(plasma_field=plasma_field,fileno=fileno)
    return(filename)




# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************


def q3d_to_3d(rundir,plasma_field,fileno,mode_max,x1_min,x1_max,nx1,x2_min,x2_max,nx2,x3_min,x3_max,nx3):
    from scipy import interpolate
    dx1=(x1_max-x1_min)/(nx1-1)
    dx2=(x2_max-x2_min)/(nx2-1)
    dx3=(x3_max-x3_min)/(nx3-1)

    x1_axis=np.arange(x1_min,x1_max+dx1,dx1)
    x2_axis=np.arange(x2_min,x2_max+dx2,dx2)
    x3_axis=np.arange(x3_min,x3_max+dx3,dx3)
    
    a = np.zeros((nx1,nx2,nx3),dtype=float)
    
    filename_out = filename_3d(rundir,plasma_field,fileno)



    x1 = osh5def.DataAxis(x1_min,x1_max, nx1, attrs={'NAME':'x1', 'LONG_NAME':'x_1', 'UNITS':'c / \omega_0'})
    x2 = osh5def.DataAxis(x2_min,x2_max, nx2, attrs={'NAME':'x2', 'LONG_NAME':'x_2', 'UNITS':'c / \omega_0'})
    x3 = osh5def.DataAxis(x3_min,x3_max, nx3, attrs={'NAME':'x3', 'LONG_NAME':'x_3', 'UNITS':'c / \omega_0'})



    # More attributes associated with the data/simulation. Again no need to worry about the details.
    data_attrs = {'UNITS': osh5def.OSUnits('m_e c \omega_0 / e'), 'NAME': plasma_field, 'LONG_NAME': plasma_field}
    run_attrs = {'NOTE': 'parameters about this simulation are stored here', 'TIME UNITS': '1/\omega_0',
            'XMAX':np.array([1., 15.]), 'XMIN':np.array([0., 10.])}



    # Now "wrap" the numpy array into osh5def.H5Data. Note that the data and the axes are consistent and are in fortran ordering
    b = osh5def.H5Data(a, timestamp='123456', data_attrs=data_attrs, run_attrs=run_attrs, axes=[x1, x2, x3])
    

    
    # I am doing mode 0 outside of the loop
    
    fname_re=filename_re(rundir,plasma_field,0,fileno)
    # DEBUG
    print(fname_re)
    # DEBUG
    data_re = osh5io.read_h5(fname_re)
    print(data_re.shape)
    print(data_re.axes[1].ax.shape)
    print(data_re.axes[0].ax.shape)
    
    func_re = interpolate.interp2d(data_re.axes[1].ax,data_re.axes[0].ax,data_re,kind='cubic')
    
    for i1 in range(0,nx1):
        for i2 in range(0,nx2):
            for i3 in range(0,nx3):
                z = x1_axis[i1]
                x = x3_axis[i3]
                y = x2_axis[i2]
                
                r = np.sqrt(x*x+y*y)
                # if r != 0:
                #     cos_th = x/r
                #     sin_th = y/r
                # else:
                #     cos_th = 1
                #     sin_th = 0
                a[i1,i2,i3]=a[i1,i2,i3]+func_re(z,r)
                
    for i_mode in range(1,mode_max+1):
        fname_re=filename_re(rundir,plasma_field,i_mode,fileno)
        fname_im=filename_im(rundir,plasma_field,i_mode,fileno)

        # DEBUG
        print(fname_re)
        # DEBUG
        if(plasma_field =='e2' or plasma_field == 'e3'):
            if (plasma_field == 'e2'):
                field_comp = 'e3'
            else:
                field_comp = 'e2'
              
            data_re_self=osh5io.read_h5(filename_re(rundir,plasma_field,i_mode,fileno))
            data_im_self=osh5io.read_h5(filename_im(rundir,plasma_field,i_mode,fileno))
            
            data_re_comp=osh5io.read_h5(filename_re(rundir,field_comp,i_mode,fileno))
            data_im_comp=osh5io.read_h5(filename_im(rundir,field_comp,i_mode,fileno))
            
        else:
            data_re=osh5io.read_h5(filename_re(rundir,plasma_field,i_mode,fileno))
            data_im=osh5io.read_h5(filename_im(rundir,plasma_field,i_mode,fileno))
            func_re = interpolate.interp2d(data_re.axes[1].ax,data_re.axes[0].ax,data_re,kind='cubic')
            func_im = interpolate.interp2d(data_im.axes[1].ax,data_im.axes[0].ax,data_im,kind='cubic')
 
        for i1 in range(0,nx1):
            for i2 in range(0,nx2):
                for i3 in range(0,nx3):
                    z = x1_axis[i1]
                    x = x3_axis[i3]
                    y = x2_axis[i2]
                    
                    r = np.sqrt(x*x+y*y)
                    
                    if r > 0.000001:
                        cos_th = x/r
                        sin_th = y/r
                    else:
                        cos_th = 1
                        sin_th = 0
                    # start the recursion relation to evaluate cos(n*theta) and sin(n_theta)  
                    sin_n=sin_th
                    cos_n=cos_th
                    for int_mode in range(2,i_mode+1):
                        temp_s=sin_n
                        temp_c=cos_n
                        cos_n=temp_c*cos_th-temp_s*sin_th
                        sin_n=temp_s*cos_th+temp_c*sin_th
                    #
                    # here we perform the addition of the N-th mode
                    # to the data in 3D
                    #
                    a[i1,i2,i3]=a[i1,i2,i3]+func_re(z,r)*cos_n-func_im(z,r)*sin_n
                        
 

    osh5io.write_h5(b,filename=filename_out)
    
                
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************
# ******************************************************************************************************

    
