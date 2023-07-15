from h5_utilities import *
import matplotlib.pyplot as plt
import sys
import glob
import numpy


argc=len(sys.argv)
if(argc < 3):
    print('Usage: python 1d_combine.py DIRNAME DIRNAME2 OUTNAME')
    sys.exit()

dirname=sys.argv[1]
dirname2=sys.argv[2]
outfilename = sys.argv[3]

filelist=sorted(glob.glob(dirname+'/*.h5'))
filelist2=sorted(glob.glob(dirname2+'/*.h5'))
total_time=len(filelist)

h5_filename=filelist[1]
h5_data=read_hdf(h5_filename)
array_dims=h5_data.shape
nx=array_dims[0]
ny=array_dims[1]
time_step=h5_data.run_attributes['TIME'][0]
h5_output=hdf_data()
h5_output.shape=[total_time,ny]
print('nx='+repr(nx))
print('ny='+repr(ny))
print('time_step='+repr(time_step))
print('total_time='+repr(total_time))
h5_output.data=numpy.zeros((total_time,ny))
h5_output.axes=[data_basic_axis(0,h5_data.axes[0].axis_min,h5_data.axes[0].axis_max,ny),
data_basic_axis(1,0.0,(time_step*total_time-1),total_time)]
h5_output.run_attributes['TIME']=0.0
# h5_output.run_attributes['UNITS']=h5_data.run_attributes['UNITS']
h5_output.axes[0].attributes['LONG_NAME']=h5_data.axes[0].attributes['LONG_NAME']
h5_output.axes[0].attributes['UNITS']=h5_data.axes[0].attributes['UNITS']
h5_output.axes[1].attributes['LONG_NAME']='TIME'
h5_output.axes[1].attributes['UNITS']='1/\omega_p'

file_number=0
for ix,h5_filename in enumerate(filelist):
  print(h5_filename)
  h5_data=read_hdf(h5_filename)
  h5_data_2=read_hdf(filelist2[ix])
  temp=numpy.sum(h5_data.data,axis=0)/nx
  temp2=numpy.sum(h5_data_2.data,axis=0)/nx
  h5_output.data[file_number,1:ny]=temp[1:ny]-temp2[1:ny]
  file_number+=1


# print 'before write'
# print outfilename
write_hdf(h5_output,outfilename)
# print 'after write'


