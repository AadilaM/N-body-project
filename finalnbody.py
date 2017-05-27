
"""
N-body Project
Aadila Moola and Salma Khan

Written in Python 3

"""


import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation

class NbodySolver:
    def __init__(self,npart=150,soft=0.001 , G=1.0, m=1, dt=0.1): #__init__ written by Salma
        self.dict={}
        self.dict['n']=npart
        self.dict['G']=G
        self.dict['soft']=soft
        self.dict['m']=m
        self.dict['dt']=dt
        self.x=np.random.randn(self.dict['n'])
        self.y=np.random.randn(self.dict['n'])
        self.xmin, self.ymin, self.xmax, self.ymax= round(min(self.x)),round(min(self.y)),round(max(self.x)),round(max(self.y))
        self.rho=np.zeros([abs(self.ymin)+self.ymax+ 1,  abs(self.xmin)+self.xmax+1])
        self.vx=np.zeros(self.dict['n'])
        self.vy=np.zeros(self.dict['n'])          
    
        
    #We consider each element of self.rho as a grid point. 
    #In get_density we assign each particle to a grid point. 
    #Self.rho then gives the mass of the particles at each grid point.  
        
    def get_density(self): #written by Aadila
    
    
        for i in range(0,self.dict['n']): 
            xindex=int(round(self.x[i]))
            yindex=int(round(self.y[i]))
            self.rho[yindex, xindex]+=1
        self.rho=self.rho*self.dict['m']
        return self.rho
        
    def get_r(self): #written by Aadila - gives distance of each grid point from origin 
        xplus= np.arange(0,self.xmax+1)
        xminus=np.arange(self.xmin,0)
        xtotal=np.concatenate((xplus,xminus)) #array of the x co-ordinates of grid points, matching rho
    
        yplus= np.arange(0,self.ymax+1)
        yminus=np.arange(self.ymin,0)
        ytotal=np.concatenate((yplus,yminus))#array of y co-ordinates of the grid points, matching rho
    
        rsq=self.rho.copy() #creating an array of distances corresponding to entries in rho
        for i in range (0,xtotal.size):
            for j in range (0,ytotal.size):
                rsq[j,i]=xtotal[i]**2+ytotal[j]**2
        soft=self.dict['soft']**2
        rsq[rsq<soft]=soft
        r=np.sqrt(rsq)
        return r 
        
    def get_potential(self): #written by Salma
        r=self.get_r()
        pot=-self.dict['G']*self.get_density()/r
        return pot 
        
    def total_potential(self): #written by Salma- uses convolution to get an array of total potential at each grid point
        rho_ft=np.fft.fft(self.get_density())
        pot_ft=np.fft.fft(self.get_potential())
        totalrho=np.fft.ifft(rho_ft*pot_ft).real
        return totalrho
        
    def get_force(self): #written by Salma- gets the force on each grid point
        self.fx=self.rho.copy()
        self.fy=0*self.fx  
        potential=self.total_potential()
        fy =-1*np.gradient(potential,axis=0)
        fx =1*np.gradient(potential,axis=1)
        return fy, fx
        
    def interpolate_force(self): #written by Aadila- interpolates force from grid points to particles
        fy,fx=self.get_force() 
        fypart=self.vy.copy()
        fxpart=fypart.copy()
        for i in range (0,fx.size):
            xindex=round(self.x[i])
            yindex=round(self.y[i])
            fxpart[i]=fx[yindex,xindex]
            fypart[i]=fy[yindex,xindex]
        return fypart, fxpart   
            
     
    def evolve(self): #written by Salma- updates position and velocities using timestep
        self.x+=self.vx*self.dict['dt']
        self.y+=self.vy*self.dict['dt']
        self.fy, self.fx = self.interpolate_force()
        self.vx+=self.fx*self.dict['dt']
        self.vy+=self.fy*self.dict['dt']
        kinetic=0.5*np.sum(self.dict['m']*(self.vx**2+self.vy**2))
        return self.total_potential()+kinetic         
     
if __name__=='__main__': #Adapted from lecture 6 code
    plt.ion()
    n=150
    oversamp=5
    part=NbodySolver (m=1.0/n,npart=n,dt=0.1/oversamp)
    plt.plot(part.x,part.y,'*')
    plt.show()
    


    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
    line, = ax.plot([], [], '*', lw=2)

  
    def animate_points(crud):
        global part,line,oversamp
        for ii in range(oversamp):
            energy=part.evolve()
        print (energy)
        line.set_data(part.x,part.y)
       
        
    ani = animation.FuncAnimation(fig, animate_points, np.arange(100),
                              interval=200, blit=False)
    plt.show()

             

        
