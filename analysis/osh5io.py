#!/usr/bin/env python

"""osh5io.py: Disk IO for the OSIRIS HDF5 data."""

__author__ = "Han Wen"
__copyright__ = "Copyright 2018, PICKSC"
__credits__ = ["Adam Tableman", "Frank Tsung", "Thamine Dalichaouch"]
__license__ = "GPLv2"
__version__ = "0.1"
__maintainer__ = "Han Wen"
__email__ = "hanwen@ucla.edu"
__status__ = "Development"


import h5py
import os
import numpy as np
from osh5def import H5Data, fn_rule, DataAxis, OSUnits


def read_h5(filename, path=None, axis_name="AXIS/AXIS"):
    """
    HDF reader for Osiris/Visxd compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf('e1-000006.h5')      # diag_data is a subclass of numpy.ndarray with extra attributes

            print(diag_data)                          # print the meta data
            print(diag_data.view(numpy.ndarray))      # print the raw data
            print(diag_data.shape)                    # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])        # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)                # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print(diag_data.axes[0].attrs['UNITS'])   # prints units of X-axis
            list(diag_data.axes[0].attrs)             # lists all variables of the X-axis

            diag_data[slice(3)]
                print(rw.view(np.ndarray))

    We will convert all byte strings stored in the h5 file to strings which are easier to deal with when writing codes
    see also write_h5() function in this file

    """
    fname = filename if not path else path + '/' + filename
    data_file = h5py.File(fname, 'r')

    n_data = scan_hdf5_file_for_main_data_array(data_file)

    timestamp, name, run_attrs, data_attrs, axes, data_bundle= '', '', {}, {}, [], []
    try:
        timestamp = fn_rule.findall(os.path.basename(filename))[0]
    except IndexError:
        timestamp = '000000'

    axis_number = 1
    while True:
        try:
            # try to open up another AXIS object in the HDF's attribute directory
            #  (they are named /AXIS/AXIS1, /AXIS/AXIS2, /AXIS/AXIS3 ...)
            axis_to_look_for = axis_name + str(axis_number)
            axis = data_file[axis_to_look_for]
            # convert byte string attributes to string
            attrs = {}
            for k, v in axis.attrs.items():
                try:
                    attrs[k] = v[0].decode('utf-8') if isinstance(v[0], bytes) else v
                except IndexError:
                    attrs[k] = v.decode('utf-8') if isinstance(v, bytes) else v

            axis_min = axis[0]
            axis_max = axis[-1]
            axis_numberpoints = n_data[0].shape[-axis_number]

            data_axis = DataAxis(axis_min, axis_max, axis_numberpoints, attrs=attrs)
            axes.insert(0, data_axis)
        except KeyError:
            break
        axis_number += 1

    # we need a loop here primarily (I think) for n_ene_bin phasespace data
    for the_data_hdf_object in n_data:
        name = the_data_hdf_object.name[1:]  # ignore the beginning '/'

        # now read in attributes of the ROOT of the hdf5..
        #   there's lots of good info there. strip out the array if value is a string

        for key, value in data_file.attrs.items():
            try:
                run_attrs[key] = value[0].decode('utf-8') if isinstance(value[0], bytes) else value
            except IndexError:
                run_attrs[key] = value.decode('utf-8') if isinstance(value, bytes) else value
        # attach attributes assigned to the data array to
        #    the H5Data.data_attrs object, remove trivial dimension before assignment
        for key, value in the_data_hdf_object.attrs.items():
            try:
                data_attrs[key] = value[0].decode('utf-8') if isinstance(value[0], bytes) else value
            except IndexError:
                data_attrs[key] = value.decode('utf-8') if isinstance(value, bytes) else value

        # convert unit string to osunit object
        try:
            data_attrs['UNITS'] = OSUnits(data_attrs['UNITS'])
        except KeyError:
            data_attrs['UNITS'] = OSUnits('a.u.')
        data_attrs['NAME'] = name

        # data_bundle.data = the_data_hdf_object[()]
        data_bundle.append(H5Data(the_data_hdf_object, timestamp=timestamp,
                                  data_attrs=data_attrs, run_attrs=run_attrs, axes=axes))
    data_file.close()
    if len(data_bundle) == 1:
        return data_bundle[0]
    else:
        return data_bundle


def read_h5_openpmd(filename, path=None):
    """
    HDF reader for OpenPMD compatible HDF files... This will slurp in the data
    and the attributes that describe the data (e.g. title, units, scale).

    Usage:
            diag_data = read_hdf_openpmd('EandB000006.h5')      # diag_data is a subclass of numpy.ndarray with extra attributes

            print(diag_data)                          # print the meta data
            print(diag_data.view(numpy.ndarray))      # print the raw data
            print(diag_data.shape)                    # prints the dimension of the raw data
            print(diag_data.run_attrs['TIME'])        # prints the simulation time associated with the hdf5 file
            diag_data.data_attrs['UNITS']             # print units of the dataset points
            list(diag_data.data_attrs)                # lists all attributes related to the data array
            list(diag_data.run_attrs)                 # lists all attributes related to the run
            print(diag_data.axes[0].attrs['UNITS'])   # prints units of X-axis
            list(diag_data.axes[0].attrs)             # lists all variables of the X-axis

            diag_data[slice(3)]
                print(rw.view(np.ndarray))

    We will convert all byte strings stored in the h5 file to strings which are easier to deal with when writing codes
    see also write_h5() function in this file

    """
    fname = filename if not path else path + '/' + filename
    with h5py.File(fname, 'r') as data_file:

        try:
            timestamp = fn_rule.findall(os.path.basename(filename))[0]
        except IndexError:
            timestamp = '00000000'

        basePath = data_file.attrs['basePath'].decode('utf-8').replace('%T', timestamp)
        meshPath = basePath + data_file.attrs['meshesPath'].decode('utf-8')

        run_attrs = {k.upper(): v for k, v in data_file[basePath].attrs.items()}
        run_attrs.setdefault('TIME UNITS', r'1 / \omega_p')

        # read field data
        lname_dict, fldl = {'E1': 'E_x', 'E2': 'E_y', 'E3': 'E_z',
                            'B1': 'B_x', 'B2': 'B_y', 'B3': 'B_z',
                            'jx': 'J_x', 'jy': 'J_y', 'jz': 'J_z', 'rho': r'\roh'}, {}
        # k is the field label and v is the field dataset
        for k, v in data_file[meshPath].items():
            # openPMD doesn't enforce attrs that are required in OSIRIS dataset
            data_attrs, dflt_ax_unit = \
                {'UNITS': OSUnits(r'm_e c \omega_p e^{-1} '),
                 'LONG_NAME': lname_dict.get(k, k), 'NAME': k}, r'c \omega_p^{-1}'
            data_attrs.update({ia: va for ia, va in v.attrs.items()})

            ax_label, ax_off, g_spacing, ax_pos, unitsi = \
                data_attrs.pop('axisLabels'), data_attrs.pop('gridGlobalOffset'), \
                data_attrs.pop('gridSpacing'), data_attrs.pop('position'), data_attrs.pop('unitSI')
            ax_min = (ax_off + ax_pos * g_spacing) * unitsi
            ax_max = ax_min + v.shape * g_spacing * unitsi

            # prepare the axes data
            axes = []
            for aln, an, amax, amin, anp in zip(ax_label,ax_label,
                                                ax_max, ax_min, v.shape):
                ax_attrs = {'LONG_NAME': aln.decode('utf-8'),
                            'NAME': an.decode('utf-8'), 'UNITS': dflt_ax_unit}
                data_axis = DataAxis(amin, amax, anp, attrs=ax_attrs)
                axes.append(data_axis)

            fldl[k] = H5Data(v[()], timestamp=timestamp, data_attrs=data_attrs,
                             run_attrs=run_attrs, axes=axes)
    return fldl


def scan_hdf5_file_for_main_data_array(h5file):
    res = []
    for k, v in h5file.items():
        if isinstance(v, h5py.Dataset):
            res.append(h5file[k])
    if not res:
        raise Exception('Main data array not found')
    return res


def write_h5(data, filename=None, path=None, dataset_name=None, overwrite=True, axis_name=None):
    """
    Usage:
        write(diag_data, '/path/to/filename.h5')    # writes out Visxd compatible HDF5 data.

    Since h5 format does not support python strings, we will convert all string data (units, names etc)
    to bytes strings before writing.

    see also read_h5() function in this file

    """
    if isinstance(data, H5Data):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = H5Data(data)
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data_object = H5Data(np.array(data))
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or somehitng that can go in a numy array")

    # now let's make the H5Data() compatible with VisXd and such...
    # take care of the NAME attribute.
    if dataset_name is not None:
        current_name_attr = dataset_name
    elif data_object.name:
        current_name_attr = data_object.name
    else:
        current_name_attr = "Data"

    fname = path if path else ''
    if filename is not None:
        fname += filename
    elif data_object.timestamp is not None:
        fname += current_name_attr + '-' + data_object.timestamp + '.h5'
    else:
        raise Exception("You did not specify a filename!!!")
    if os.path.isfile(fname):
        if overwrite:
            os.remove(fname)
        else:
            c = 1
            while os.path.isfile(fname[:-3]+'.copy'+str(c)+'.h5'):
                c += 1
            fname = fname[:-3]+'.copy'+str(c)+'.h5'
    h5file = h5py.File(fname)

    # now put the data in a group called this...
    h5dataset = h5file.create_dataset(current_name_attr, data_object.shape, data=data_object.view(np.ndarray))
    # these are required.. so make defaults ones...
    h5dataset.attrs['UNITS'], h5dataset.attrs['LONG_NAME'] = np.array([b'']), np.array([b''])
    # convert osunit class back to ascii
    data_attrs = data_object.data_attrs.copy()
    try:
        data_attrs['UNITS'] = np.array([str(data_object.data_attrs['UNITS']).encode('utf-8')])
    except:
        data_attrs['UNITS'] = np.array([b'a.u.'])
    # copy over any values we have in the 'H5Data' object;
    for key, value in data_attrs.items():
        h5dataset.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
    # these are required so we make defaults..
    h5file.attrs['DT'] = [1.0]
    h5file.attrs['ITER'] = [0]
    h5file.attrs['MOVE C'] = [0]
    h5file.attrs['PERIODIC'] = [0]
    h5file.attrs['TIME'] = [0.0]
    h5file.attrs['TIME UNITS'] = [b'']
    h5file.attrs['TYPE'] = [b'grid']
    h5file.attrs['XMIN'] = [0.0]
    h5file.attrs['XMAX'] = [0.0]
    # now make defaults/copy over the attributes in the root of the hdf5
    for key, value in data_object.run_attrs.items():
        h5file.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value

    number_axis_objects_we_need = len(data_object.axes)
    # now go through and set/create our axes HDF entries.
    if not axis_name:
        axis_name = "AXIS/AXIS"
    for i in range(0, number_axis_objects_we_need):
        _axis_name = axis_name + str(number_axis_objects_we_need - i)
        if _axis_name not in h5file:
            axis_data = h5file.create_dataset(_axis_name, (2,), 'float64')
        else:
            axis_data = h5file[_axis_name]

        # set the extent to the data we have...
        axis_data[0] = data_object.axes[i].min
        axis_data[1] = data_object.axes[i].max

        # fill in any values we have stored in the Axis object
        for key, value in data_object.axes[i].attrs.items():
            if key == 'UNITS':
                try:
                    axis_data.attrs['UNITS'] = np.array([str(data_object.axes[i].attrs['UNITS']).encode('utf-8')])
                except:
                    axis_data.attrs['UNITS'] = np.array([b'a.u.'])
            else:
                axis_data.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
    h5file.close()


def write_h5_openpmd(data, filename=None, path=None, dataset_name=None, overwrite=True, axis_name=None,
    time_to_si=1.0, length_to_si=1.0, data_to_si=1.0 ):
    """
    Usage:
        write_h5_openpmd(diag_data, '/path/to/filename.h5')    # writes out Visxd compatible HDF5 data.

    Since h5 format does not support python strings, we will convert all string data (units, names etc)
    to bytes strings before writing.

    see also read_h5() function in this file

    """
    if isinstance(data, H5Data):
        data_object = data
    elif isinstance(data, np.ndarray):
        data_object = H5Data(data)
    else:
        try:  # maybe it's something we can wrap in a numpy array
            data_object = H5Data(np.array(data))
        except:
            raise Exception(
                "Invalid data type.. we need a 'hdf5_data', numpy array, or something that can go in a numy array")

    # now let's make the H5Data() compatible with VisXd and such...
    # take care of the NAME attribute.
    if dataset_name is not None:
        current_name_attr = dataset_name
    elif data_object.name:
        current_name_attr = data_object.name
    else:
        current_name_attr = "Data"

    fname = path if path else ''
    if filename is not None:
        fname += filename
    elif data_object.timestamp is not None:
        fname += current_name_attr + '-' + data_object.timestamp + '.h5'
    else:
        raise Exception("You did not specify a filename!!!")
    if os.path.isfile(fname):
        if overwrite:
            os.remove(fname)
        else:
            c = 1
            while os.path.isfile(fname[:-3]+'.copy'+str(c)+'.h5'):
                c += 1
            fname = fname[:-3]+'.copy'+str(c)+'.h5'
    h5file = h5py.File(fname)

    # now put the data in a group called this...
 #   h5dataset = h5file.create_dataset(current_name_attr, data_object.shape, data=data_object.view(np.ndarray))
    # these are required.. so make defaults ones...
 #   h5dataset.attrs['UNITS'], h5dataset.attrs['LONG_NAME'] = np.array([b'']), np.array([b''])
    # convert osunit class back to ascii
    data_attrs = data_object.data_attrs.copy()
    try:
        data_attrs['UNITS'] = np.array([str(data_object.data_attrs['UNITS']).encode('utf-8')])
    except:
        data_attrs['UNITS'] = np.array([b'a.u.'])
    # copy over any values we have in the 'H5Data' object;
#    for key, value in data_attrs.items():
#        h5dataset.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
    # these are required so we make defaults..
    h5file.attrs['DT'] = [1.0]
    h5file.attrs['ITER'] = [0]
    h5file.attrs['MOVE C'] = [0]
    h5file.attrs['PERIODIC'] = [0]
    h5file.attrs['TIME'] = [0.0]
    h5file.attrs['TIME UNITS'] = [b'']
    h5file.attrs['TYPE'] = [b'grid']
    h5file.attrs['XMIN'] = [0.0]
    h5file.attrs['XMAX'] = [0.0]
    h5file.attrs['openPMD'] = '1.0.0'
    h5file.attrs['openPMDextension'] = 0
    h5file.attrs['iterationEncoding'] = 'fileBased' 
    h5file.attrs['basePath']='/data/%T'
    h5file.attrs['meshesPath']='mesh/'
    h5file.attrs['particlesPath']= 'particles/' 
    # now make defaults/copy over the attributes in the root of the hdf5

    baseid = h5file.create_group("data")
    iterid = baseid.create_group(str(data.run_attrs['ITER'][0]))


    meshid = iterid.create_group("mesh")
    datasetid = meshid.create_dataset(data_attrs['NAME'], data_object.shape, data=data_object.view(np.ndarray) )

   # h5dataset = datasetid.create_dataset(current_name_attr, data_object.shape, data=data_object.view(np.ndarray))


 #   for key, value in data_object.run_attrs.items():
 #       h5file.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value



    iterid.attrs['dt'] = data.run_attrs['DT'][0]
    iterid.attrs['time'] = data.run_attrs['TIME'][0] 
    iterid.attrs['timeUnitSI'] = time_to_si


    number_axis_objects_we_need = len(data_object.axes)

    deltax= data.run_attrs['XMAX'] - data.run_attrs['XMIN']
    local_offset = np.arange(number_axis_objects_we_need, dtype = np.float32)
    local_globaloffset = np.arange(number_axis_objects_we_need, dtype = np.float64)
    local_position = np.arange(number_axis_objects_we_need, dtype = np.float32)
    local_position[0] = 0.0
    local_position[1] = 0.0

    local_gridspacing = np.arange(number_axis_objects_we_need, dtype = np.float32)

    if(number_axis_objects_we_need == 1):
        local_axislabels=[b'x1']
        deltax[0] = deltax[0]/data_object.shape[0]
        local_gridspacing=np.float32(deltax)

        local_globaloffset[0] = np.float32(0.0)

        local_offset[0]= np.float32(0.0)
 
    elif (number_axis_objects_we_need == 2):
        local_axislabels=[b'x1', b'x2']
        deltax[0] = deltax[0]/data_object.shape[0]
        deltax[1] = deltax[1]/data_object.shape[1]
        temp=deltax[0]
        deltax[0]=deltax[1]
        deltax[1]=temp
        local_gridspacing=np.float32(deltax)

        local_globaloffset[0] = np.float32(0.0)
        local_globaloffset[1] = np.float32(0.0)

        local_offset[0]= np.float32(0.0)
        local_offset[1]= np.float32(0.0)

    else:
        local_axislabels=[b'x1',b'x2',b'x3']
        deltax[0] = deltax[0]/data_object.shape[0]
        deltax[1] = deltax[1]/data_object.shape[1]
        deltax[2] = deltax[2]/data_object.shape[2]
        temp=deltax[0]
        deltax[0]=deltax[2]
        deltax[2]=temp
        local_gridspacing=np.float32(deltax)

        local_globaloffset[0] = np.float32(0.0)
        local_globaloffset[1] = np.float32(0.0)
        local_globaloffset[2] = np.float32(0.0)

        local_offset[0]= np.float32(0.0)
        local_offset[1]= np.float32(0.0)
        local_offset[2]= np.float32(0.0)


     
    datasetid.attrs['dataOrder'] = 'F'
    datasetid.attrs['geometry'] = 'cartesian'
    datasetid.attrs['geometryParameters'] =  'cartesian'
    datasetid.attrs['axisLabels'] = local_axislabels
    datasetid.attrs['gridUnitSI'] = np.float64(length_to_si)
    datasetid.attrs['unitSI'] = np.float64(data_to_si)
    datasetid.attrs['position'] = local_position
    datasetid.attrs['gridSpacing'] = local_gridspacing
    datasetid.attrs['gridGlobalOffset'] = local_globaloffset


    # # now go through and set/create our axes HDF entries.
    # if not axis_name:
    #     axis_name = "AXIS/AXIS"
    # for i in range(0, number_axis_objects_we_need):
    #     _axis_name = axis_name + str(number_axis_objects_we_need - i)
    #     if _axis_name not in h5file:
    #         axis_data = h5file.create_dataset(_axis_name, (2,), 'float64')
    #     else:
    #         axis_data = h5file[_axis_name]

    #     # set the extent to the data we have...
    #     axis_data[0] = data_object.axes[i].min
    #     axis_data[1] = data_object.axes[i].max

    #     # fill in any values we have stored in the Axis object
    #     for key, value in data_object.axes[i].attrs.items():
    #         if key == 'UNITS':
    #             try:
    #                 axis_data.attrs['UNITS'] = np.array([str(data_object.axes[i].attrs['UNITS']).encode('utf-8')])
    #             except:
    #                 axis_data.attrs['UNITS'] = np.array([b'a.u.'])
    #         else:
    #             axis_data.attrs[key] = np.array([value.encode('utf-8')]) if isinstance(value, str) else value
    h5file.close()



if __name__ == '__main__':
    import osh5utils as ut
    a = np.arange(6.0).reshape(2, 3)
    ax, ay = DataAxis(0, 3, 3, attrs={'UNITS': '1 / \omega_p'}), DataAxis(10, 11, 2, attrs={'UNITS': 'c / \omega_p'})
    da = {'UNITS': 'n_0', 'NAME': 'test', }
    h5d = H5Data(a, timestamp='123456', data_attrs=da, axes=[ay, ax])
    write_h5(h5d, './test-123456.h5')
    rw = read_h5('./test-123456.h5')
    h5d = read_h5('./test-123456.h5')  # read from file to get all default attrs
    print("rw is h5d: ", rw is h5d, '\n')
    print(repr(rw))

    # let's read/write a few times and see if there are mutations to the data
    # you should also diff the output h5 files
    for i in range(5):
        write_h5(rw, './test' + str(i) + '-123456.h5')
        rw = read_h5('./test' + str(i) + '-123456.h5')
        assert (rw == a).all()
        for axrw, axh5d in zip(rw.axes, h5d.axes):
            assert axrw.attrs == axh5d.attrs
            assert (axrw == axh5d).all()
        assert h5d.timestamp == rw.timestamp
        assert h5d.name == rw.name
        assert h5d.data_attrs == rw.data_attrs
        assert h5d.run_attrs == rw.run_attrs
        print('checking: ', i+1, 'pass completed')

    # test some other functionaries
    print('\n meta info of rw: ', rw)
    print('\nunit of rw is ', rw.data_attrs['UNITS'])
    rw **= 3
    print('unit of rw^3 is ', rw.data_attrs['UNITS'])
    print('contents of rw^3: \n', rw.view(np.ndarray))


