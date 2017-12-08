from h5_utilities import *
import matplotlib.pyplot as plt
import sys
import glob
import numpy
import IPython.display

argc=len(sys.argv)
if(argc < 3):
	print('Usage: python 1d_combine.py DIRNAME OUTNAME')
	sys.exit()

#def combine(dirname, outfilename):
dirname=sys.argv[1]
outfilename = sys.argv[2]

filelist=sorted(glob.glob(dirname+'/*.h5'))
total_time=len(filelist)

h5_filename=filelist[1]
h5_data = read_hdf(h5_filename)
array_dims=h5_data.shape
nx=array_dims[0]
# ny=array_dims[1]
time_step=h5_data.run_attributes['TIME'][0]
h5_output=hdf_data()
h5_output.shape=[total_time,nx]
print('nx='+repr(nx))
# print('ny='+repr(ny))
print('time_step='+repr(time_step))
print('total_time='+repr(total_time))
h5_output.data=numpy.zeros((total_time,nx))
h5_output.axes=[data_basic_axis(0,h5_data.axes[0].axis_min,h5_data.axes[0].axis_max,nx),
data_basic_axis(1,0.0,(time_step*total_time-1),total_time)]
h5_output.run_attributes['TIME']=0.0
# h5_output.run_attributes['UNITS']=h5_data.run_attributes['UNITS']
h5_output.axes[0].attributes['LONG_NAME']=h5_data.axes[0].attributes['LONG_NAME']
h5_output.axes[0].attributes['UNITS']=h5_data.axes[0].attributes['UNITS']
h5_output.axes[1].attributes['LONG_NAME']='TIME'
h5_output.axes[1].attributes['UNITS']='1/\omega_p'

file_number=0
for h5_filename in filelist:
  print(h5_filename)
  h5_data=read_hdf(h5_filename)
  #  temp=numpy.sum(h5_data.data,axis=0)/nx
  h5_output.data[file_number,1:nx]=h5_data.data[1:nx]
  file_number+=1


# print 'before write'
# print outfilename
write_hdf(h5_output,outfilename)
# print 'after write'
print('combine_h5_util_1d.py completed normally')
