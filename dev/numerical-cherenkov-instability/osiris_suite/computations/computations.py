from osiris_suite import OsirisDataContainer 
import numpy as np 


def compute_fft_2d( data ) : 
	
	fft = np.fft.fftn( data ) 
	fft = np.abs( fft ) 

	# k2_axis = np.fft.fftfreq( data.shape[0] )
	# k1_axis = np.fft.fftfreq( data.shape[1] )

	# k1_axis = np.fft.fftshift( k1_axis )
	# k2_axis = np.fft.fftshift( k2_axis )
	
	fft = np.fft.fftshift( fft ) 

	# return fft, ( k1_axis, k2_axis )
	return fft, ( [-0.5, 0.5], [-0.5, 0.5] )



def compute_lorentz_transformation( osdata, gamma, 
								e_leaves, b_leaves, j_leaves,
								phase_space_leaves, 
								load = 1, save = 0 ) : 

	'''
		apply lorentz boost along the direction of the first leaf
		the leaves must be specified in a right-handed coordinate system
		e.g. ( e3, e1, e2 ) would transform along z direction in cartesian
		coordinates 
	'''

	boosted_data = OsirisDataContainer() 

	# check if the boost gamma is compatible with the unboosted mesh
