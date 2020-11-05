import h5py

recursive = 1 
step = 2 


# print all datasets 

def scan_hdf5( group ) : 
	
	for key, val in group.items():

		if isinstance(val, h5py.Dataset):
			# print(' ' * num_spaces + '---> ' + v.name)
			print( val.name )

		elif isinstance( val, h5py.Group ) :
			scan_hdf5( val ) 