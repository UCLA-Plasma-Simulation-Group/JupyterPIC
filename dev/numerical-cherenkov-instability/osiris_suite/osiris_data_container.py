import numpy as np
import os 
import glob
import sys
import h5py
import collections 
from pprint import pprint  
import re 


from .helper_classes import AttrDict, RecursiveAttrDict, OsirisSuiteError
from .input_deck_manager import InputDeckManager



class H5FileManager( object ) : 

	def __init__( self, path = None, data_key = None ) : 

		self.path = path 
		self.file = None 
		self.data_key = data_key 


	def load( self ) : 

		try : 
			self.file = h5py.File( self.path, 'r')

		except OSError : 
			print( 'OSError: unable to open %s' % files[i] )


	def unload( self ) : 

		self.file.close() 
		self.file = None 


	# pull out top level dataset if it exists
	# if the file has multiple top level datasets, then the desired 
	# data must be manually extracted by using a similar function.
	@property
	def data( self ) : 
		# return self.file[ self.data_key ]
		num_datasets = 0 
		dataset = None 

		if not self.file : 
			OsirisSuiteError( 'ERROR: attempted to access data before loading file. \
				Make sure to call H5FileManager.load() before using.')

		for key, val in self.file.items() : 

			if isinstance( val, h5py.Dataset ) : 
				dataset = key 
				num_datasets += 1 

		if num_datasets > 1 : 
			raise OsirisSuiteError( 'ERROR: multiple top-level datasets found. \
				the .data attribute cannot be used. Desired dataset \
				must manaully be extracted. This data structure still works, but \
				you can\'t use the data attribute.' )

		if num_datasets == 0 : 
			raise OsirisSuiteError( 'ERROR: no top-level dataset found for file %s. \
				The file may not be formatted according to usual OSIRIS conventions.' % self.path )

		return self.file[ dataset ]


	@property 
	def axes( self ) : 

		if not self.file : 
			OsirisSuiteError( 'ERROR: attempted to access data before loading file. \
				Make sure to call H5FileManager.load() before using.')

		axes = self.file[ '/AXIS/']
		keys = sorted( axes.keys() )

		return np.array( [ axes[k] for k in keys ] )


	def unpack( self ) : 

		self.load() 

		data = self.data
		axes = self.axes 

		self.unload

		return data, axes 


	def __bool__( self ) : 
		return self.file is None 


	def __str__( self ) : 
		ret = '' 
		ret += 'path: %s\n' % self.path 

		if self.file : 
			ret += 'status: file loaded\n'
			ret += 'axes: %s\n' % str( self.axes ) 
			ret += 'data: %s' % str( self.data ) 
		else : 
			ret += 'status: file not loaded'

		return ret 




MS_PREFIX   = '/MS'
HIST_PREFIX = '/HIST'
TIMINGS_PREFIX = '/TIMINGS'



# class OsirisData( h5py.File ) : 




class OsirisDataContainer( object ) : 
	'''

	'''


	def __init__( self, data_path = None, input_deck_path = None, 
				load_whole_file = False,
				load_empty_directories = False,
				silent = False , index_by_timestep = False ) : 
		'''
			data_path: path to parent directory containing all OSIRIS data (i.e. the directory)
				containing the OSIRIS output directory MS).

			load_whole_file: if nonzero, this will make the entire hdf5 get loaded for each file
				instead of returning a handle to the hdf5 file accessed at the data. 
				this does not add much overhead, but if you are only planning to access the data
				and none of the metadata you don't need to access the whole file. i include the option
				because it is occasionally useful to inspect the other data.  

			load_empty_directories: if True, will assume that any empty directories were supposed to contain 
				hdf5 files, and hence will create a stub for these. if False, the directories will be ignored.

			silent: if True, will prevent debug info from being printed. 

			index_by_timestep: if True, data is stored with index = t_0, ..., t_N where t_i are the N timesteps 
				output for this data in the simulation. if False, the data is stored with index = 0 ... N independent 
				of the timesteps. in either case the timesteps variable stores the values t_0 ... t_N.
		'''
	
		if not os.path.exists( data_path ) : 
			raise OSError( 'Error: the specified path does not exist: %s' % data_path )

		self.data_path = data_path 
		self.input_deck_path = input_deck_path
		self.load_whole_file = load_whole_file
		self.load_empty_directories = load_empty_directories
		self.silent = silent 
		self.index_by_timestep = index_by_timestep

		# data structures
		self.data = RecursiveAttrDict() 
		self.computations = AttrDict() 
		self.has_empty_dir = False
		self.input_deck = None 

		self.data.ms = AttrDict() 
		self.data.hist = AttrDict()
		self.timings = dict()

		if self.data_path is not None : 
			self.load_ms_tree() 

		if self.input_deck_path is not None : 
			self.input_deck = InputDeckManager( self.input_deck_path )

		# track all the indices that have data loaded 
		# self.loaded_indices = set()
		# self._keys_at_prefix = {} 





	def load_indices( self, indices = None, timesteps = None, keys = None, prune = 1 ) : 
		
		self.load_ms( indices, timesteps, keys ) 

		if self.has_empty_dir : 
			if not self.silent : 
				print( 'INFO: pruning empty data branches' )
			self.data.prune()	






	def unload_indices( self, indices = None, timesteps = None, keys = None ) : 
		... 



	def load_ms_tree( self,  indices = None, timesteps = None, keys = None ) : 
		
		ms_path = self.data_path + MS_PREFIX 

		# pprint( list( os.walk (ms_path)) )

		if not os.path.exists( ms_path ) : 
			raise OSError( 'ERROR: the MS directory does not exist: %s' % ms_path)

		self.recursively_load_dir( self.data.ms, ms_path, indices )

		# self.recursively_load_dir( self.data.hist, hist_path )




	def recursively_load_dir( self, parent_dict, directory, indices ) : 

		curpath, subdir_names, files = next( os.walk( directory ) ) 

		if len( subdir_names ) > 0 : 
			for subdir_name in subdir_names : 
				# basename = os.path.basename( os.path.normpath( subdir ) ).lower() 
				# parent_dict[ basename ] = AttrDict() 
				subdir = os.path.join( directory, subdir_name )
				key = subdir_name.lower().replace( '-', '_' )
				parent_dict[ key ] = AttrDict() 
				self.recursively_load_dir( parent_dict[ key ], subdir, indices )

		# otherwise load files 
		else : 
			if len( files ) > 0 : 
				self.collect_h5_paths( parent_dict, directory, indices ) 

			else : 	
				self.has_empty_dir = True
				if not self.silent : 
					print( 'WARNING: found empty directory: %s' % directory )

				if self.load_empty_directories : 
					self.collect_h5_paths( parent_dict, directory, indices )

				# else : 
				# 	curkey = 
				# 	del self.parent_dict[ ] 



	def collect_h5_paths( self, parent_dict, directory, indices, slice_ = None ) : 

		files = sorted( glob.glob( directory + '/*.h5' ) )
		timesteps = [ get_timestep( fname ) for fname in files ]

		varname = os.path.basename( os.path.normpath( directory ) )

		parent_dict[ 'timesteps' ] = timesteps 
		parent_dict[ 'file_managers' ] = [ None for i in range( len( timesteps ) ) ]

		# enable negative indexing in the style of numpy arrays
		if indices is None : 
			indices = np.arange( len( timesteps ) )

		indices_tmp = list( indices )[:]
		for i in range( len( indices ) ) : 
			if indices[i] < 0 : 
				indices[i] += len( timesteps ) 

		for i in range( len( files ) ) : 

			if indices is not None and i not in indices : 
				continue
			
			# print( 'info: loading timestep ' + str( timesteps[i]))

			# data = h5py.File( files[i], 'r')
			file_mgr = H5FileManager( path = files[i], data_key = varname )

			parent_dict.file_managers[ i ] = file_mgr 



	# return the first path found to data with the name leaf_name
	# e.g. if leaf_name = 'e2' is passed, then self.data.ms.e2 will be returned 
	# not implemented 
	def find_subtree( self, subtree_name, current_dict = None ) : 
		
		return self.data.find_subtree( subtree_name )



	def load_timings( self ) : 

		timings_path = self.data_path + TIMINGS_PREFIX
		self.timings = None 



	def load_hist( self ) : 
		
		hist_path = self.data_path + HIST_PREFIX

		# files = 
		# curpath, subdir_names, files = next( os.walk( hist_path ) ) 

		files = glob.glob( hist_path + '/*' )

		for file in files : 

			varname = os.path.basename( os.path.normpath( file ) )

			with open( file, 'r' ) as f : 

				# find the last comment and break 
				while( 1 ) : 

					line = f.readline() 

					if line[0] != '!' : 

						# convert multi space delimeters to tabs
						line = line.strip() 
						line = line.replace( '  ', '*' )

						# get rid of periods
						line = line.replace( '.', '' )

						# get rid of single spaces in header (e.g. "total energy")
						line = line.replace( ' ', '' ) 

						# header = line.split( sep = '*' ) 
						header = re.split( '\*+', line )

						break 
								
				data = np.loadtxt( f ) 

				# print( len( header ) ) 

				self.data.hist[ varname ] = AttrDict() 

				for i in range ( len( header ) ) : 

					self.data.hist[ varname ][ header[i] ] = data[:,i]






	def __str__( self ) : 
		return str( self.data ) 

	# def 

		


		
			
# helper functions
def idx_to_str( idx ) :
	return '%06d' % idx 



def get_timestep( os_h5_fname ) : 
	left = os_h5_fname.rfind( '-' ) + 1 
	right = os_h5_fname.rfind( '.' )

	timestep = int( os_h5_fname[ left : right ] )
	# print( timestep )
	return timestep 



def get_num_files( output_path ) :	
	num_files = len( glob.glob( output_path ) ) 
	return num_files

						







