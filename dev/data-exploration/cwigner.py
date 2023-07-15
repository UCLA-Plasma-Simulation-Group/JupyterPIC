import numpy as np
# the rwigner function takes a 1D numpy array and makes a wigner transform. 
#
def cwigner(data):
    nx=data.shape[0]
    nxh = (nx+1)/2
    temp=np.zeros((nx,nx),dtype=np.csingle)
    wdata=np.zeros((nx,nx),dtype=np.csingle)
    # temp1d=np.zeros(nx,dtype=np.csingle)
    dataplus=np.zeros(nx,dtype=np.csingle)
    dataminus=np.zeros(nx,dtype=np.csingle)
    temp[:,0]=data[:]
    
    for j in range(1,nx):
        dataplus=np.roll(data,j)
        dataminus=np.roll(data,-j)
        temp[:,j] = dataplus[:] * np.conj(dataminus[:])
        
    
    for i in range(nx):
        temp1d=np.fft.fft(temp[:,i])
        wdata[i,:]=temp1d[:]
        
    return wdata
    
    
    
