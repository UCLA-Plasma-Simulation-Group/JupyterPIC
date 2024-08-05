import h5py as h5
from matplotlib.pyplot import axis
import numpy as np
import math

def Gamma(p):
    return  math.sqrt(1 + p**2)
def Velocity(pi,p):
    return pi / Gamma(p)

def find_nearest_index(array,value):
    
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx 

def group_dat(d):
    order=['xi','r','p','pz','pr','Fr','Fz','z','t','vr','vz']
    return [d[i] for i in order]

class Tracks:
    def __init__(self,dirname = 'linear',file_id = 3):
        name=('00000'+str(int(file_id)))[-6:]
        den_path = [dirname +'/MS/DENSITY/H2elec/charge/charge-H2elec-'+name+'.h5']
        self.field_obj = ['e1','e2','e3','b1','b2','b3','psi']
        field_path = [dirname +'/MS/FLD/'+opt+'/'+opt+'-00000'+str(file_id)+'.h5' for opt in self.field_obj]
        self.path = field_path+den_path
        self.dat={}
        self.axis={}
        self.load_data()
        self.track_dat=['r','pr','pz','p','vr','vz','t','z','xi','Fr','Fz']
        self.para={i:[] for i in self.track_dat}

    def set_init_para(self,r_0,xi_0,vph,dtw, prinit = 0, pzinit =0, t0 =0):
        self.para={i:[] for i in self.track_dat}
        self.para["r"].append(r_0) # position in c/w_p
        self.para['pr'].append(prinit) # momentum in m_e c
        self.para['pz'].append(pzinit)
        p=np.sqrt(prinit**2+pzinit**2)
        self.para['p'].append(p)
        self.para['xi'].append(xi_0)
        self.para['vr'].append(prinit/Gamma(p)) # velocity in c
        self.para['t'].append(t0) # start time in 1/w_p
        self.dt = dtw # time step in 1/w_p
        self.vph=vph
        self.para['z'].append(xi_0 + t0*vph)
  
        self.para['vz'].append(pzinit/Gamma(p))
        self.para['Fz'].append(0)
        self.para['Fr'].append(0)


    def motion(self):
        xiDex = find_nearest_index(self.axis['xi'], self.para['xi'][-1])
        rDex = find_nearest_index(self.axis['r'], self.para['r'][-1])
        #p = math.sqrt(self.para['pr'][-1]**2 + self.para['pz'][-1]**2)
        #vr = Velocity(self.para['pr'][-1],self.para['p'][-1])
        #vz = Velocity(self.para['pz'][-1],self.para['p'][-1])
        self.para['Fz'].append(self.EField(rDex, xiDex, 1) + self.BForce(rDex,xiDex,self.para['vz'][-1],self.para['vr'][-1],1))
        self.para['Fr'].append(self.EField(rDex, xiDex, 2) + self.BForce(rDex,xiDex,self.para['vz'][-1],self.para['vr'][-1],2))
 
        npz = self.para['pz'][-1] + self.para['Fz'][-1]* self.dt
        npr = self.para['pr'][-1] + self.para['Fr'][-1]* self.dt
        np = math.sqrt(npr**2 + npz**2)
        self.para['pz'].append(npz)
        self.para['pr'].append(npr)
        self.para['p'].append(np)
   
        vzn = Velocity(npz,np)
        vrn = Velocity(npr,np)
        self.para['vz'].append(vzn)
        self.para['vr'].append(vrn)
        zn = vzn * self.dt+self.para['z'][-1]
        rn = vrn * self.dt+self.para['r'][-1]
        self.para['z'].append(zn)
        self.para['r'].append(rn)
        t = self.dt+self.para['t'][-1]
        self.para['t'].append(t)
        xin = zn - self.vph*t
        self.para['xi'].append(xin)

    def run(self,step,energy=100):
        n=0
        
        
        cond=[]
        while n<step and sum(cond)==0:
            self.motion()
            n=n+1
            cond=[self.para['r'][-1] > self.axis['r'].max(),
            self.para['r'][-1] < self.axis['r'].min(),
            self.para['xi'][-1] < self.axis['xi'].min(),
            self.para['xi'][-1] > self.axis['xi'].max(),
            Gamma(self.para['p'][-1]) > self.para['p'][0] + energy]
            
        
            
    def result(self):
        return group_dat(self.para)


    def getdata(self,id,axis=False):
        f = h5.File(self.path[id],"r")
        datasetNames = [n for n in f.keys()] #Two Datasets: AXIS and e2
        field = datasetNames[-1]
        Field_dat = f[field][:].astype(float)
        if axis:
            a1_bounds = f['AXIS']['AXIS1']
            a2_bounds = f['AXIS']['AXIS2']

            xi_dat = np.linspace(0,a1_bounds[1]-a1_bounds[0] ,len(Field_dat[0]))
            r_dat = np.linspace(a2_bounds[0],a2_bounds[1],len(Field_dat))
            return r_dat,xi_dat
        return Field_dat

    def load_data(self):
        for i in range(len(self.field_obj)):
            self.dat[self.field_obj[i]]=self.getdata(id=i)
        self.axis['r'],self.axis['xi']=self.getdata(id=-1,axis=True)

    def EField(self,r,xi,axis):
        return -self.dat[self.field_obj[axis-1]][r,xi]

    def BForce(self,r,xi,v1,v2,axis):
        BField =  self.dat[self.field_obj[5]][r,xi]
        if axis == 1:
            return -1.0 * v2 * BField
        else:
            return 1.0 * v1 * BField

'''
track=Tracks()
track.set_init_para(r_0=0,xi_0=10,vph=0.994987,dtw=0.4,pzinit=0)
track.run(step=2500)
print((track.para['xi']))
'''


        
    #data = [xi_dat,r_dat,p_dat,pz_dat,pr_dat,F_dat, Fz_dat, z_dat, t_dat]
 
      #trajectory.append([xi_dat,r_dat,p_dat,pz_dat,pr_dat,F_dat, Fz_dat, z_dat, t_dat])
 
    #del r_dat, xi_dat, z_dat, t_dat, pr_dat,pz_dat,F_dat, Fz_dat
    #return data
    