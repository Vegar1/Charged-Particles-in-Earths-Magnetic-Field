'''Particle Paths in Earths Magnetic Field'''

import numpy as np
from mayavi import mlab

#options
save = 0 #set to 0 if you don't want to save the path to file
plot = 1 #set to 0 if you don't want to create a plot
 
#constants
k = 8.095*(10**15)	#found by setting the value of the dipole field equal to the measured field strength at the surface of the earth
rE = 6.378*(10**6)
q,m = 1.602*(10**-19),1.672*(10**-27)
qom = q/m

#initial conditions
V = 4*10**5

B = np.array([0,0,0],dtype=float)	    #B-field
a = np.array([0,0,0],dtype=float)	    #acceleration
v = np.array([-V,0,0],dtype=float)	    #velocity
P = np.array([7*rE,0*rE,1*rE])		    #position

#save the positions to an array
X = np.array(P[0])
Y = np.array(P[1])
Z = np.array(P[2])

#rotation around the y-axis, 0.314 radians = 18 degrees
theta = 0.314
c = np.cos(theta)
s = np.sin(theta)

i = 0
while i < 50000:
    i=i+1		
    r = np.sqrt(P[0]**2 + P[1]**2 + P[2]**2)  
    t = 0.2*((r/(10*rE))**3)	        #stepsize
    '''
    The first block is technically unnecessary, but sans rotational adjustment it's easier to see what's going on. Using Euler's method we calculate the field at a given position, P, and then calculate the acceleration, a, on the particle at P based on the speed, V, and the acceleration. The acceleration is then normed to conform to the principle of energy conservation, to minimize error.
    '''
    if theta == 0:
        B[0] = ((k * (P[0]*P[2])) / r**5)
        B[1] = ((k * (P[2]*P[1])) / r**5)
        B[2] = ((k * ((P[2]**2) - (r**2)/3)) / r**5)	
    else: 
        B[0] = (c*((k * ((c*P[0]-s*P[2])*(s*P[0]+c*P[2]))) / r**5) + s*((k * (((s*P[0]+c*P[2])**2) - (r**2)/3)) / r**5))
        B[1] = ((k * ((s*P[0]+c*P[2])*P[1])) / r**5)
        B[2] = (-s*((k * ((c*P[0]-s*P[2])*(s*P[0]+c*P[2]))) / r**5) + c*((k * (((s*P[0]+c*P[2])**2) - (r**2)/3)) / r**5)) 
		
    a = qom*np.cross(v,B)	

    '''
    Since magnetic forces do no work, we would expect the absolute value of the velocity to not change. However, a naive implementation of Euler's method does not respect that condition, so we decompose v and renormalize the component of v perpendicular to B.
    '''
        		
    vB = np.dot(v,B)*B/np.dot(B,B)		#projection of v on B     
    v_B = v - vB                        #rejection of v on B	
    v_rescaled = (a*t + v_B)*np.linalg.norm(v_B)/np.linalg.norm(a*t + v_B)   
    v = v_rescaled + vB		
    P = P + v*t + a*t*t/2
    #makes sure the particle doesn't crash and burn
    if np.linalg.norm(P) < rE:
        break	 

    #adds the new position to the array
    X = np.append(X,P[0])
    Y = np.append(Y,P[1])
    Z = np.append(Z,P[2])	

#saves the path to a file
if save == 1:
    var_theta = "{:.3f}".format(theta)
    var_i = "{:.5f}".format(i)
    var_x = "{:.2f}".format(X[0])
    var_y = "{:.2f}".format(Y[0])
    var_z = "{:.2f}".format(Z[0])
    variables = (var_theta + '_' + var_i + '_' 
    + var_x + '_' + var_y + '_' + var_z)

    np.savetxt('i_theta_x_y_z{0}.txt'.format(variables),[X,Y,Z])

#plots particle trajectory
if plot == 1:    
    mlab.plot3d(X,Y,Z,tube_radius=5000)	
    #adds axis of the magnetic field
    mlab.plot3d(np.array([-s*4*rE,s*4*rE]),
    np.array([0,0]),
    np.array([-c*4*rE,c*4*rE]),
    tube_radius=30000,color=(1, 0, 0))
    #adds Tellus to the plot
    mlab.points3d(0, 0, 0,
    resolution=64,scale_factor=2*rE,color=(0, 0, 1))

mlab.show()
