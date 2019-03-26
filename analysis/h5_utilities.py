import matplotlib
import matplotlib.cm
import matplotlib.colors
import h5py
import numpy as np
from numpy import *
from pylab import *

import os

def plotme(hdf_data, data = None, **kwargs):

    data_to_use = hdf_data.data
    if data is not None:
        data_to_use = data

    if len(data_to_use.shape) == 1:
        plot_object =  plot(hdf_data.axes[1].get_axis_points(), data_to_use)
        xlabel("%s     %s" % (hdf_data.axes[0].attributes['LONG_NAME'][0], (hdf_data.axes[0].attributes['UNITS'] )[0]))
        ylabel("%s       %s"% ( hdf_data.data_attributes['LONG_NAME'], (hdf_data.data_attributes['UNITS'])) )
        return plot_object

    if len(data_to_use.shape) == 2:
        extent_stuff = [hdf_data.axes[0].axis_min, hdf_data.axes[0].axis_max, hdf_data.axes[1].axis_min,
                        hdf_data.axes[1].axis_max]
        plot_object = imshow(data_to_use, extent=extent_stuff, aspect='auto',origin='lower',**kwargs)
        cb=colorbar(plot_object) # original cmap='Rainbow'
#         cb.set_label("%s \n %s"% ( hdf_data.data_attributes['LONG_NAME'], hdf_data.data_attributes['UNITS']) )
#         xlabel("%s        %s" % (hdf_data.axes[0].attributes['LONG_NAME'][0], (hdf_data.axes[0].attributes['UNITS'])[0] ))
#         ylabel("%s        %s" % (hdf_data.axes[1].attributes['LONG_NAME'][0], (hdf_data.axes[1].attributes['UNITS'][0] )))
        #ylabel("%s \n %s"% ( hdf_data.data_attributes['LONG_NAME'], hdf_data.data_attributes['UNITS']) )

def plotmetranspose(hdf_data, data = None, **kwargs):

    data_to_use = hdf_data.data
    if data is not None:
        data_to_use = data

    if len(data_to_use.shape) == 1:
        plot_object =  plot(hdf_data.axes[0].get_axis_points(), data_to_use)
        xlabel("%s     %s" % (hdf_data.axes[1].attributes['LONG_NAME'][1], (hdf_data.axes[1].attributes['UNITS'] )[1]))
        ylabel("%s       %s"% ( hdf_data.data_attributes['LONG_NAME'], (hdf_data.data_attributes['UNITS'])) )
        return plot_object

    if len(data_to_use.shape) == 2:
        extent_stuff = [hdf_data.axes[1].axis_min, hdf_data.axes[1].axis_max, hdf_data.axes[0].axis_min,
                        hdf_data.axes[0].axis_max]
        plot_object = imshow(data_to_use, extent=extent_stuff, aspect='auto',origin='lower',**kwargs)
        cb=colorbar(plot_object) # original cmap='Rainbow'
#         cb.set_label("%s \n %s"% ( hdf_data.data_attributes['LONG_NAME'], hdf_data.data_attributes['UNITS']) )
#         xlabel("%s        %s" % (hdf_data.axes[0].attributes['LONG_NAME'][0], (hdf_data.axes[0].attributes['UNITS'])[0] ))
#         ylabel("%s        %s" % (hdf_data.axes[1].attributes['LONG_NAME'][0], (hdf_data.axes[1].attributes['UNITS'][0] )))
        #ylabel("%s \n %s"% ( hdf_data.data_attributes['LONG_NAME'], hdf_data.data_attributes['UNITS']) )

def math_string(input):
    try:
        input = input[0]
    except:
        pass
        input = str(input)
        v = input
#    v = r"$" + v + "$"
#        print v
    return v
class hdf_data:

    def __init__(self):
        self.filename = None
        self.axes = []
        self.data = None
        self.data_attributes = {}
        self.run_attributes = {}

    def clone(self):
        out = hdf_data()
        out.filename = filename

        for axis in self.axes:
            out_axis = axis.clone()

        for key,value in self.data_attributes.items():
            out.data_attributes [key] = value

        for key,value in self.run_attributes.items():
            out.run_attributes [key] = value
    """
    def register_slice(self, data, list_indices_removed):
        temp_list = []
        for axis_index in range(0, len(self.axes)):
            if axis_index not in list_indices_removed:
                temp_list.append(item)
    """

    #full_expression.append(x_slice)
    def slice(self, x3=None, x2=None, x1=None, copy=False):
        slice_index_array = []
        target_obj = self

        temp_axes = list(self.axes)
        for axis in temp_axes:
            #print "HI!!!!"
            selection = slice(None, None, None)
            # print "Gonna slice axis %d" % axis.axis_number
            if axis.axis_number == 1:
                selection = self.__slice_dim(x3, self.data, 3)
            if axis.axis_number == 2:
                selection = self.__slice_dim(x2, self.data, 2)
            if axis.axis_number == 3:
                selection = self.__slice_dim(x1, self.data, 1)

            slice_index_array.append(selection)

        # print slice_index_array
        new_data = self.data[slice_index_array]
        #
        #
        self.data = None
        self.data = new_data
        self.shape = new_data.shape

    def __slice_dim(self, indices, data, axis_direction):

        if indices == None:
            return slice(None, None, None)

        # a slicer tha tis just an interger
        #    means take a slice using that value.
        #    so remove the coresponding axis.
        if isinstance( indices, ( int, long ) ):
            if(len(self.axes) > 1):
                self.__remove_axis(axis_direction)
            return indices

        array_slice = None
        #we accept lots of different specifications for the indecies.. let's sort it out.
        if(indices != None):
            # now see if a python list was passed in.
            #if so, make it into a slicer
            try:
                size = len(indices)
                if size == 0:
                    array_slice = slice(None, None, None)
                elif size == 1:
                    array_slice = slice(indices[0], None, None)
                elif size == 2:
                    array_slice = slice(indices[0], indices[1], None)
                elif size == 3:
                    array_slice = slice(indices[0], indices[1], indices[2])
            except:
                # Well, what ever is there, just try to use it like a slicer
                #        if it works, then go with it.
                array_slice = indices

            try:
                # These next lines just test that we have a 'slicer'
                #        (or functional equivilent). If these fileds are not
                #        there, then an exception will be thrown.
                temp = array_slice.start
                temp = array_slice.stop
                temp = array_slice.end

                #------START LOGIC---------------
                # by this time, we have processed all different input types
                # into a standard form: a slicer object named 'indices'

                # a slicer with start=None and end=None means take the whole axis as-is.
                if array_slice.start == None and array_slice.stop == None:
                    return array_slice

                axis = self.get_axis(axis_direction)

                new_start_index = 0
                new_stop_index = len(data)
                if array_slice.start != None:
                    new_start_index = array_slice.start
                if array_slice.stop != None:
                    new_stop_index = array_slice.stop


                # now update the axis's data to reflect the new slice start/stop.
                # update the Python side.
                axis.axis_min = (axis.increment*new_start_index) + axis.axis_min
                axis.axis_max = (axis.increment*new_stop_index) + axis.axis_min
                self.axis_numberpoints = new_stop_index - new_stop_index

                #update to dictionary 'attributues' will happen in the
                # 'write_hdf' function.,.. just to keep all HDF action localised to
                # isolated routines.
                return array_slice

                #TODO: handle the Elipsis object..

                # also can use test[ix_(a_dim,b_dim,c_dim)] but not quite flexable enough?
            except:
                # broken slicer..
                raise Exception("Invalid indices for array slice")

        return slice(None, None, None)

    # return None if axis is not present.
    def get_axis(self, axis_index):
        for (i,axis) in enumerate(self.axes):
            if axis.axis_number==axis_index:
                return axis
        return None

    def __remove_axis(self, axis_index):
        for (i,axis) in enumerate(self.axes):
            if axis.axis_number==axis_index:
                del self.axes[i]
        pass
    def __axis_exists(self, axis_index):
        for (i,axis) in enumerate(self.axes):
            if axis.axis_number==axis_index:
                return True
        return False

class data_basic_axis:
    def __init__(self, axis_number, axis_min, axis_max, axis_numberpoints):

        self.axis_number            = axis_number
        self.axis_min                = axis_min
        self.axis_max                = axis_max
        self.axis_numberpoints    = axis_numberpoints
        self.increment            = (self.axis_max-self.axis_min)/self.axis_numberpoints
        self.attributes                = {}
    def clone(self):
        out = data_basic_axis()
        out.axis_number = self.axis_number
        out.axis_min = self.axis_min
        out.axis_max = self.axis_max
        out.axis_numberpoints = self.axis_numberpoints
        out.increment = self.increment
        for key,value in self.attributes.items():
            out.attributes [key] = value

    def get_axis_points(self):
        return np.arange(self.axis_min, self.axis_max, self.increment)

def read_hdf(filename):
    """
    HDF reader for Osiris/Visxd compatable HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf('e1-000006.h5')

            data = diag_data.data                        # gets the raw data
            print diag_data.data.shape                    # prints the dimension of the raw data
            print diag_data.run_attributes['TIME']        # prints the simulation time associated with the hdf5 file
            diag_data.data_attributes['UNITS']            # print units of the dataset points
            list(diag_data.data_attributes)                # lists all variables in 'data_attributes'
            list(diag_data.run_attributes)                # lists all vairables in 'run_attributes'
            print diag_data.axes[0].attributes['UNITS'] # prints units of  X-axis
            list(diag_data.axes[0].attributes['UNITS']) # lists all variables of the X-axis

            diag_data.slice( x=34, y=(10,30) )
            diag_data.slice(x=3)

            diag_data.write(diag_data, 'filename.h5')    # writes out Visxd compatiable HDF5 data.


    (See bottom of file 'hdf.py' for more techincal information.)

    """


    data_file1 = h5py.File(filename, 'r')

    the_data_hdf_object = scan_hdf5_file_for_main_data_array(data_file1)
    dim = len(the_data_hdf_object.shape)

    data_bundle = hdf_data()
    data_bundle.filename = filename
    data_bundle.dim = len(the_data_hdf_object.shape)
    data_bundle.shape = list(the_data_hdf_object.shape)

    # now read in attributes of the ROOT of the hdf5.. t
    #    there's lots of good info there.
    for key,value in data_file1.attrs.items():
        data_bundle.run_attributes[key] = value
        setattr(data_bundle, str(key), value)

    # attach attributes assigned to the data array to
    #    the hdf_data.data_attrs object
    for key, value in the_data_hdf_object.attrs.items():
        data_bundle.data_attributes[key] = value

    axis_number = 1

    while True:
        try:
            # try to open up another AXIS object in the HDF's attribute directory
            #  (they are named /AXIS/AXIS1, /AXIS/AXIS2, /AXIS/AXIS3 ...)
            axis_to_look_for = "/AXIS/AXIS" + str(axis_number)
            axis =    data_file1[axis_to_look_for]
            axis_data = axis[:]
            axis_min = axis_data.item(0)
            axis_max = axis_data.item(1)
            axis_numberpoints = the_data_hdf_object.shape[axis_number-1]

            data_axis = data_basic_axis(axis_number, axis_min, axis_max, axis_numberpoints)
            data_bundle.axes.append(data_axis)
            # get the attributes for the JUST ADDED AXIS
            for key, value in axis.attrs.items():
                data_axis.attributes[key] = value
        except:
            break
        axis_number += 1

    #TODO: probabaly better way to do this..
    if dim == 1:
        data_bundle.data = the_data_hdf_object[:]
    elif dim == 2:
        data_bundle.data = the_data_hdf_object[:][:]
    elif dim == 3:
        data_bundle.data = the_data_hdf_object[:][:][:]
    else:
        raise ValueException("You attempted to read in an Osiris diagnostic which had data of dimension greater then 3.. cant do that yet.")

    data_file1.close()
    return data_bundle


def scan_hdf5_file_for_main_data_array(file):
    datasetName = ""
    for k,v in file.items():
        if isinstance(v, h5py.highlevel.Dataset):
            datasetName = k
            break
    return file[datasetName]


if __name__=="__main__":

    data = read_hdf('x3x2x1-s1-000090.h5')

#    print data.data_attributes
#    print dir(data.axes[0])
#    print
#    print dir(data)



def write_hdf(data, filename, dataset_name = None, write_data = True):

    if(os.path.isfile(filename)):
        os.remove(filename)
    try:
        dim = len(data.axes)
        data_object = data
        #if we get here, its prolly a 'hdf_data' object...
        # thats what we want.. so stop and wait..
        pass
    except:
        try:
            # This is maybe a numpy array
            type = data.dtype
            data_object = hdf_data()
            data_object.data = data

        except:
            try:
                #maybe it's something we can wrap in a numpy array
                data = np.array(data)
                data_object = hdf_data()
                data_object.data = data
            except:
                raise Exception("Invalid data type.. we need a 'hdf5_data', numpy array, or somehitng that can go in a numy array")

    # now let's make the hdf_data() compatible with VisXd and such...
    # take care of the NAME attribute.
    if dataset_name != None:
        current_NAME_attr = dataset_name
    else:
        try:
            current_NAME_attr = data_object.run_attributes['NAME'][0]
        except:
            current_NAME_attr = "Data"

    # now put the data in a group called this...


    if(filename != None):
        file = h5py.File(filename)
        data_object.filename = filename
    elif data_object.filename != None:
        file = h5py.File(filename)
    else:
        raise Exception("You did not specify a filename!!!")

    #    print current_NAME_attr
    h5dataset = file.create_dataset(current_NAME_attr, data_object.shape, data=data_object.data)
    # these are required.. so make defaults ones...
    h5dataset.attrs['UNITS'] = ''
    h5dataset.attrs['LONG_NAME'] = ''
    #copy over any values we have in the 'hdf_data' object;'s
    #print data_object.data_attributes
    for key,value in data_object.data_attributes.items():
        h5dataset.attrs[key] = value

    # these are required so we make defaults..
    file.attrs['DT'] = 1.0
    file.attrs['ITER'] = 0
    file.attrs['MOVE C'] = [0,0,0]
    file.attrs['PERIODIC'] = [0,0,0]
    file.attrs['TIME'] = 0.0
    file.attrs['TIME UNITS'] = ''
    file.attrs['TYPE'] = 'grid'
    file.attrs['XMIN'] = [0.0, 0.0, 0.0, 0.0]
    file.attrs['XMAX'] = [1.0, 1.0, 1.0, 1.0]
    # now make defaults/copy over the attributes in the root of the hdf5
    for key,value in data_object.run_attributes.items():
        file.attrs[key] = value
    # in order to fill in XMIN/XMAX, let's use the values we have in the axes objects.
    xmin = [0.0,0.0,0.0,0.0]
    xmax = [1.0,1.0,1.0,1.0]
    #TODO: find out if x,y,z have semantic meaning or is it just the order of the axes in the list.
    for i,axis in enumerate(data_object.axes):
        xmin[i] =  data_object.axes[i].axis_min
        xmax[i] =  data_object.axes[i].axis_max
    file.attrs['XMIN']    = xmin
    file.attrs['XMAX']    = xmax

    # now create the axis objects....
    # first see if the AXIS group (folder) exists..
    if 'AXIS' not in file:
        grp = file.create_group("AXIS")
    # now go though the group and remove any extra AXISx arrays
    number_axis_object_present =  len(file['AXIS'].keys())
    number_axis_objects_we_need = len(data_object.axes)
    for i in range(0, number_axis_object_present):
        axis_name = "AXIS/AXIS%d" % (i+1)
        if axis_name in file:
            if i<number_axis_objects_we_need:
                pass
            else:
                del file[axis_name]
    # now go through and set/create our axes HDF entries.
    for i in range(0, number_axis_objects_we_need):
        axis_name = "AXIS/AXIS%d" % (i+1)
        if axis_name not in file:
            axis_data = file.create_dataset(axis_name, (2,), 'float64')
        else:
            axis_data = file[axis_data]

        # set the extent to the data we have...
        axis_data[0] = data_object.axes[i].axis_min
        axis_data[1] = data_object.axes[i].axis_max

        # now make attributes for axis that are required..
        axis_data.attrs['UNITS'] = ""
        axis_data.attrs['LONG_NAME'] = ""
        axis_data.attrs['TYPE'] = ""
        axis_data.attrs['NAME'] = ""
        # fill in any values we have storedd in the Axis object
        for key,value in data_object.axes[i].attributes.items():
            axis_data.attrs[key] = value


    if write_data:
        file.close()

#----------------------------------------------------------------------------------------------------------
#                    Setup maps etc...
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
idl_13_r=[0,4,9,13,18,22,27,31,36,40,45,50,54,58,61,64,68,69,72,74,77,79,80,82,83,85,84,86,87,88,86,87,87,87,85,84,84,84,83,79,78,77,76,71,70,68,66,60,58,55,53,46,43,40,
            36,33,25,21,16,12,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,8,12,21,25,29,33,42,46,51,55,63,67,72,76,80,89,93,97,101,110,114,119,123,131,135,140,144,153,157,161,
            165,169,178,182,187,191,199,203,208,212,221,225,229,233,242,246,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
idl_13_g=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,8,16,21,25,29,38,42,46,51,55,63,67,72,76,84,89,93,97,106,110,114,
            119,127,131,135,140,144,152,157,161,165,174,178,182,187,195,199,203,208,216,220,225,229,233,242,246,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,250,242,
            238,233,229,221,216,212,208,199,195,191,187,178,174,170,165,161,153,148,144,140,131,127,123,119,110,106,102,97,89,85,80,76,72,63,59,55,51,42,38,34,29,21,
            17,12,8,0]
idl_13_b=[0,3,7,10,14,19,23,28,32,38,43,48,53,59,63,68,72,77,81,86,91,95,100,104,109,113,118,122,127,132,136,141,145,150,154,159,163,168,173,177,182,186,
            191,195,200,204,209,214,218,223,227,232,236,241,245,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,246,242,238,233,225,220,216,212,203,199,195,191,187,178,174,
            170,165,157,152,148,144,135,131,127,123,114,110,106,102,97,89,84,80,76,67,63,59,55,46,42,38,34,25,21,16,12,8,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0]


def init_colormap():
        rgb = []
        for i in range(0,len(idl_13_r)):
            rgb.append((float(idl_13_r[i])/255.0,float(idl_13_g[i])/255.0,float(idl_13_b[i])/255.0))
        color_map = matplotlib.colors.ListedColormap(rgb, name = 'Rainbow')
        color_map.set_under(rgb[0])
        color_map.set_over(rgb[-1])
        matplotlib.cm.register_cmap(name='Rainbow', cmap=color_map)
init_colormap()

"""

    A Minimum is assumed about the HDF5 structure. What is assumed is:
        0.    There is an attribute in the root of the hdf5 named 'NAME'
                (you would access it with: h5py_file['NAME']). This contains the
                array entry of the that holds the main data.. so say h5py_file['NAME']) = 'p1x1'.
                Then the main data array will be under the name 'p1x1'.
                You could get with:       h5py_file['p1x1'].value

        1.        In the HDF, the is a group (a group is just a folder in MAC osx) named 'AXIS'.
                Inside that folder sits the info about each axis in the HDF dataset. For 1D data,
                there is 1 entry: AXIS/AXIS1. If you get the array data here
                using:          h5py_file['AXIS/AXIS1'].value
                It will be a 1D array with only 2 values; these are the min/max of the axis when plotting.
                Each axis also had some attributes. The 4 usual ones are TYPE,UNITS,NAME,LONG_NAME (type
                is linear or log etc).. Others are string that can be used for axis labels. In 2D there are
                two entries under AXIS folder and 3 for 3D (h5py_file['AXIS/AXIS1'], h5py_file['AXIS/AXIS2'],
                h5py_file['AXIS/AXIS3'])
        2. We need 1 more 'axis' (no matter in 1d, 2d, or 3d). It is this access that will decribe the actual
                quantity being measued (E.g. the veritcal axis in most 1-D plots... the height or color of a region in
                a 2D etc.. This indo is in attributes attched to the main data array we found in Step 0.
                It has 2 attributes: UNITS and LONG NAME for labeling the data when plotted.
        3. The root has lots of attributes. It has stuff about time:
                    ITER = iteration data is showing. DT=step size used,
                    TIME=simualtion time data is showing, TIME_UNITS=string.. good label to put on plots.
                    "MOVE C" = [x, y, z] where 0 for a componet means that component isn'i interested
                                    and a 1 means the compent wants to/or have done this ("MOVE C" tells us moving
                                    window directions. PERODIC is same idea as "MOVE C".. just showing perodic directions.
                    "XMIN" = [x_min, y_min, z_min] and "XMAX" = [x_max, y_max, z_max]..
                    always 3 compents.
        4. ALL DATA/ATTRIBUTES ARE COPIED OUT OF HDF INTO MEMORY. Then HDF file is closed. There are other ways.. and for
            very large files that wont fit in memory, other options will have to be used that allow 'streaming' of the
            input files.

"""
