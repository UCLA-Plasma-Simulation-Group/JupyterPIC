from osiris_suite import OsirisDataContainer, InputDeckManager
from osiris_suite.plotting import *
from osiris_suite.computations import compute_fft_2d

import numpy as np 
import matplotlib.pyplot as plt 
import os 

template_input_deck = './template-input-deck.os'


class NCIInputDeckOptions( object ) : 

	def __init__( self ) :

		self.solver_type = 'yee'
		self.use_current_filter = False

		self.tmax = 0
		self.num_output_data = 0
		
		self.nx1 = 0 
		self.nx2 = 0 
		self.dx1 = 0 
		self.dx2 = 0 
		self.dt = 0

		self.gamma = 0
		self.density = 0
		
		# for yee solver: 
		self.yee_order = 0
		
		# bump solver options 
		self.bump_order = 0 
		self.bump_n_coef = 0
		self.klower = 0
		self.kupper = 0 
		self.dk = 0

		# filter options 
		self.filter_limit = 0
		self.filter_width = 0 
		self.n_damp_cell = 0 


def generate_input_deck( nci_inputdeck_options, sim_name ) : 

	x = nci_inputdeck_options

	if x.dt**2 >= x.dx1**2 + x.dx2**2 : 
		raise Exception( 'ERROR: CFL condition violated.' )

	# read deck into memory and modify it 
	deck = InputDeckManager( template_input_deck )

	# modify the resolution params 
	nx_p = np.array( [ x.nx1, x.nx2 ] )
	deck[ 'grid' ][ 'nx_p(1:2)' ] = nx_p 
	xmax = nx_p * np.array( [ x.dx1, x.dx2 ] ) 
	deck[ 'space' ][ 'xmax(1:2)' ] = xmax
	deck[ 'time_step' ][ 'dt' ] = x.dt 
	deck[ 'time' ][ 'tmax' ] = x.tmax

	# compute ndump 
	ndump_diag_emf = round( x.tmax / ( x.dt * x.num_output_data ) ) 
	deck[ 'diag_emf' ][ 'ndump_fac' ] = ndump_diag_emf

	for i in range(2) : 
		deck[ 'udist', i ][ 'ufl(1:3)' ] = np.array( [ x.gamma, 0, 0 ] )
		deck[ 'profile', i ][ 'density' ] = x.density 
	
	# handle emf solver params
	emf_solver = deck[ 'emf_solver' ]

	if x.solver_type == 'yee' : 
		emf_solver[ 'type' ] = 'standard'
		emf_solver[ 'solver_ord' ] = x.yee_order 

	elif x.solver_type == 'bump' : 
		emf_solver[ 'type' ] = 'bump'
		emf_solver[ 'solver_ord' ] = x.bump_order
		emf_solver[ 'n_coef' ] = x.bump_n_coef  	
		emf_solver[ 'kl' ] = x.klower 
		emf_solver[ 'ku' ] = x.kupper 
		emf_solver[ 'dk' ] = x.dk	

	else : 
		print( 'ERROR: unrecognized solver type' )

	# custom filters 
	if x.use_current_filter : 
		emf_solver[ 'filter_limit' ] = x.filter_limit
		emf_solver[ 'filter_width' ] = x.filter_width
		emf_solver[ 'n_damp_cell' ] = x.n_damp_cell 
		emf_solver[ 'filter_current' ] = True
		emf_solver[ 'correct_current' ] = True

	else : 
		emf_solver[ 'filter_current' ] = False
		emf_solver[ 'correct_current' ] = False

	os.makedirs( get_sim_path( sim_name ), exist_ok = True )
	#path = get_input_deck_path( sim_name )
	path = 'input.os'
	deck.write( path )	



def get_sim_path( sim ) : 
	# return './simulations/' + sim + '/'
	return sim + '/'

def get_input_deck_path( sim ) : 
	return get_sim_path(sim) + 'input.os'


from IPython.display import Video

def show_movie( sim ) : 

	path = sim + '/plots/nci-plots/nci-plots.mp4'

	Video( path, embed = True, width = 800 )



def make_movie_individual( sim ) :

	data_path = get_sim_path( sim ) 
	# input_deck_path = get_input_deck_path( sim ) 
	input_deck_path = data_path + 'osiris-input.txt'	

	osdata = OsirisDataContainer( data_path, input_deck_path = input_deck_path )
	osdata.load_hist() 
	osdata.load_timings()

	shape = (2,3)

	# example of using custom data for a plot, in this 
	# case the spatial FFT of E2 
	def fft_computation( leaf, index ) : 

		data, axes = leaf.file_managers[ index ].unpack()
		return osiris_suite.computations.compute_fft_2d( data )


	def dens_fft_computation( index ) : 
		
		leaf = osdata.data.ms.density.electrons.charge
		data, axes = leaf.file_managers[ index ].unpack()
		data = np.array( data )
		data += osdata.input_deck[ 'profile' ][ 'density' ]

		return osiris_suite.computations.compute_fft_2d( data )


	def energy_plot_data_getter( index ) : 

		data = []
		axes = []

		for key in [ 'E1', 'E2', 'E3' ] : 	
			fld_data = osdata.data.hist.fld_ene[ key ]
			fld_times = osdata.data.hist.fld_ene[ 'Time' ]
			data.append( fld_data )
			axes.append( [[ fld_times[0], fld_times[-1] ]] ) 

		electron_ke_data = osdata.data.hist.par01_ene[  'KinEnergy' ]
		electron_ke_times = osdata.data.hist.par01_ene[ 'Time' ]
		data.append( electron_ke_data )
		axes.append( [[electron_ke_times[0], electron_ke_times[-1] ]] )

		return data, axes  


	e2_plot_mgr = raw_osdata_TS2D_plot_mgr( 
		osdata.data.ms.fld.e2, 
		cmap = 'PiYG', 
		logscale = 0, 
		title = 'E2' )

	dens_plot_mgr = raw_osdata_TS2D_plot_mgr( 
		osdata.data.ms.density.electrons.charge, 
		cmap = 'PiYG', 
		logscale = 0, 
		title = 'Electron Density' )


	e2_fft_plot_mgr = PlotManager( 
		data_getter = lambda index : fft_computation( osdata.data.ms.fld.e2, index ),
		plotter = Plotter2D( cmap = 'plasma', # colorcet.m_fire, 
							 logscale = 1, 
							 title = 'FFT of E2') )

	dens_fft_plot_mgr = PlotManager( 
		data_getter = dens_fft_computation,
		plotter = Plotter2D( cmap = 'plasma', 
							 logscale = 1, 
							 title = 'FFT of $\\tilde n$') )

	#######
	p1x1_plot_mgr = raw_osdata_TS2D_plot_mgr( 
		osdata.data.ms.pha.p1x1.electrons, 
		cmap = 'inferno', 
		logscale = 0, 
		title = 'p1 vs. x1' )



	energy_plot_mgr = PlotManager( 
		data_getter = energy_plot_data_getter, 
		plotter = Plotter1D( multiple_data = True,
							 colors = 'rgbk', 
						 	 linestyles = '----',
							 labels = ['E1', 'E2', 'E3', 'Electron KE'],
							 title = 'Total energy vs. time',
							 logy = 1 ) )

	#######

	plot_mgr_arr = \
	[
		[ energy_plot_mgr, e2_plot_mgr, dens_plot_mgr ],
		[ p1x1_plot_mgr, e2_fft_plot_mgr, dens_fft_plot_mgr ],
	]



	# parameters to space the plots. fiddle with them until the plots 
	# are spaced properly on your screen. 
	subplots_adjust = (0.2, 0.5)

	# could change timesteps here. a linear 
	timesteps = osdata.data.ms.fld.e1.timesteps 

	suptitle = sim 

	savedir = get_movie_dir( sim ) 

	# when debugging, set the parameter show = 1 and keep killing the 
	# program / modifying the inputs until you like it; then set show = 0
	# to make all plots and generate the movie. 
	osiris_suite.plotting.make_TS_movie( 	osdata, timesteps, 
										 	shape, plot_mgr_arr, 
										 	suptitle = suptitle,
											global_modifier_function = None,
											subplots_adjust = subplots_adjust,
											savedir = savedir,
											# show_index = 10,
											nproc = 4,
											duration = 5 )

def get_movie_dir( sim ) : 
	return get_sim_path( sim )  + 'plots/nci-plots/'


def make_plots_group() : 
	... 


