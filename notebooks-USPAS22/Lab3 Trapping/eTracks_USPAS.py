# This is a modified version of the eTrack code developed by the SBU-PAG group.
# It tracks a single particle inside the wakefield in the co-moving frame.
# by Jiayang Yan on June 2022

import h5py as h5
import numpy as np
import math

def getdata(id,path):
    f = h5.File(path[id],"r")
    datasetNames = [n for n in f.keys()] #Two Datasets: AXIS and e2
    field = datasetNames[-1]
    Field_dat = f[field][:].astype(float)

    a1_bounds = f['AXIS']['AXIS1']
    a2_bounds = f['AXIS']['AXIS2']

    xi_dat = np.linspace(0,a1_bounds[1]-a1_bounds[0] ,len(Field_dat[0]))
    r_dat = np.linspace(a2_bounds[0],a2_bounds[1],len(Field_dat))
    return Field_dat, r_dat, xi_dat
    

def getAxes(path):
    
    den,r,xi = getdata(-1,path)
    return r, xi

def longE(path):
    Ez, a,b = getdata(0,path)
    return Ez

def transE(path):
    Er, a,b = getdata(1,path)
    return Er

def phiB(path):
    Bphi, a,b = getdata(5,path)
    return Bphi


def Gamma(p):
    return  math.sqrt(1 + p**2)
def Velocity(pi,p):
    v = pi / Gamma(p)
    return v

def find_nearest_index(array,value):
    
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx 
    
def EField(r,xi,axis,path):
    # axis = 1 refers to xi-axis (longitudinal) field
    # axis = 2 refers to r-axis (transverse) field
    r_sim,xi_sim = getAxes(path)
    Er_sim = transE(path)
    Ez_sim = longE(path)
    if axis == 2:
        xiDex = find_nearest_index(xi_sim, xi)
        rDex = find_nearest_index(r_sim, r)
        return -Er_sim[rDex,xiDex]
    elif axis == 1:
        xiDex = find_nearest_index(xi_sim, xi)
        rDex = find_nearest_index(r_sim, r)
        return -Ez_sim[rDex, xiDex]

def BForce(r,xi,v1,v2,axis,path):
    r_sim,xi_sim = getAxes(path)
    Bphi_sim = phiB(path)
    xiDex = find_nearest_index(xi_sim, xi)
    rDex = find_nearest_index(r_sim, r)
    BField =  Bphi_sim[rDex, xiDex]
    if axis == 1:
        return -1.0 * v2 * BField
    else:
        return 1.0 * v1 * BField

def Momentum(r, xi, dt, pr, pz,path):
    p = math.sqrt(pr**2 + pz**2)
    vr = Velocity(pr,p)
    vz = Velocity(pz,p)

    Fz = (EField(r, xi, 1,path) + BForce(r,xi,vz,vr,1,path))
    Fr = (EField(r, xi, 2,path) + BForce(r,xi,vz,vr,2,path))
 
    pz = pz + Fz * dt
    pr = pr + Fr * dt
    p = math.sqrt(pr**2 + pz**2)
  
    return pz, pr, p, Fr, Fz
def GetTrajectory(path,r_0,xi_0,vph,dtw,step, prinit = 0, pzinit =0, t0 =0):
   
    r_dat, z_dat, t_dat, xi_dat, vz_dat,p_dat,pz_dat,pr_dat,F_dat, Fz_dat = np.array([]),np.array([]),np.array([]), np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
    trajectories = []
    
    r_sim,xi_sim = getAxes(path)
    Er_sim = transE(path)
    Ez_sim = longE(path)
    Bphi_sim = phiB(path)

    rn = r_0 # position in c/w_p
    pr = prinit # momentum in m_e c
    pz = pzinit
    pinit = np.sqrt(prinit**2+pzinit**2)  
    p = np.sqrt(pr**2+pz**2)

    vrn = pr/Gamma(p) # velocity in c
    t = t0 # start time in 1/w_p
    dt = dtw # time step in 1/w_p

    z0 = xi_0 + t0*vph
    zn = xi_0 + t0*vph
  
    vzn = pz/Gamma(p)
    dvz = 0.0
    dvr = 0.0

    old_r = r_0 #- 1.0
    turnRad = r_0
    xin = xi_0
    

   
  #Iterate through position and time using a linear approximation
  #until the radial position begins decreasing
    i = 0 #iteration counter
  # Iterate while electron energy is under 10GeV(100 MeV)
 
    while Gamma(p) < pinit + 100 and i< int(step):
        
    #Determine Momentum and velocity at this time and position
   
        pz, pr, p, F, Fz = Momentum(rn, xin, dt, pr, pz,path)
   
        vzn = Velocity(pz,p)
        vrn = Velocity(pr,p)
 
    #Add former data points to the data lists
        r_dat = np.append(r_dat, rn)
        t_dat = np.append(t_dat, t)
        z_dat = np.append(z_dat, zn)
        vz_dat = np.append(vz_dat, vzn)
        xi_dat = np.append(xi_dat, xin)
        pr_dat = np.append(pr_dat, pr)
        pz_dat = np.append(pz_dat, pz)
        p_dat = np.append(p_dat, p)
        F_dat = np.append(F_dat,F)
        Fz_dat = np.append(Fz_dat, Fz)
    #print("z = ", zn)
        if rn > turnRad:
            turnRad = rn

    #print("vz=",vzn)
    #Add the distance traveled in dt to r, increase t by dt
        zn += vzn * dt
        rn += vrn * dt
        t += dt
        xin = zn - vph*t
        i += 1

    # Allow for crossing the beam axis
        #if rn < 0:
        #    rn = -rn
        #    pr = -pr
        if rn > r_sim.max() or rn < r_sim.min() or xin < xi_sim.min() or xin > xi_sim.max():
            
            #print("out of box!")
            
            break
    
    xiPos = xin
        
    data = [xi_dat,r_dat,p_dat,pz_dat,pr_dat,F_dat, Fz_dat, z_dat, t_dat]
 
      #trajectory.append([xi_dat,r_dat,p_dat,pz_dat,pr_dat,F_dat, Fz_dat, z_dat, t_dat])
 
    del r_dat, xi_dat, z_dat, t_dat, pr_dat,pz_dat,F_dat, Fz_dat
    return data
    
