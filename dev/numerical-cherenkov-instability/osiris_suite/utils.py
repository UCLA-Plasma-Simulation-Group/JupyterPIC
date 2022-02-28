
def check_leaf_timesteps( data_arr ) : 

	# verify timesteps 
	success = 1 

	timesteps = None 

	# timesteps = data_arr[0].timesteps
	# indices = np.arange( len( timesteps ) )

	for leaf in data_arr : 
		
		try : 

			if timesteps is None : 
				timesteps = leaf.timesteps 

			success &= np.allclose( timesteps, leaf.timesteps ) 
		except : 
			pass  

	return success
