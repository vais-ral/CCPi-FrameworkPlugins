
# This demo illustrates how CCPi 2D parallel-beam projectors can be used with
# the modular optimisation framework. The demo sets up a small 4-slice 3D test 
# case and demonstrates reconstruction using CGLS, as well as FISTA for least 
# squares and 1-norm regularisation and FBPD for 1-norm regularisation.

# First make all imports
from ccpi.framework import ImageData, AcquisitionData, ImageGeometry, \
                           AcquisitionGeometry

from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D

from ccpi.plugins.ops import CCPiProjectorSimple
from ccpi.plugins.processors import CCPiForwardProjector, CCPiBackwardProjector 

from ccpi.reconstruction.parallelbeam import alg as pbalg

import numpy as np
import matplotlib.pyplot as plt

# Set up phantom size N x N x vert by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display one slice as image.

# Image parameters
N = 128
vert = 4

# Set up image geometry
ig = ImageGeometry(voxel_num_x=N,
                   voxel_num_y=N, 
                   voxel_num_z=vert)

# Set up empty image data
Phantom = ImageData(geometry=ig,
                    dimension_labels=['horizontal_x',
                                      'horizontal_y',
                                      'vertical'])

# Populate image data by looping over and filling slices
i = 0
while i < vert:
    if vert > 1:
        x = Phantom.subset(vertical=i).array
    else:
        x = Phantom.array
    x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
    x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 0.98
    if vert > 1 :
        Phantom.fill(x, vertical=i)
    i += 1

# Display slice of phantom
if vert > 1:
    plt.imshow(Phantom.subset(vertical=0).as_array())
else:
    plt.imshow(Phantom.as_array())
plt.show()


# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam, set the width of a detector 
# pixel relative to an object pixe and the number of detector pixels.
angles_num = 20; # angles number
det_w = 1.0
det_num = N

angles = np.linspace(0,np.pi,angles_num,endpoint=False,dtype=np.float32)*180/np.pi

#center_of_rotation = Phantom.get_dimension_size('horizontal_x') / 2

ag = AcquisitionGeometry('parallel',
                         '3D',
                         angles,
                         N, 
                         det_w,
                         vert,
                         det_w)

# CCPi operator using image and acquisition geometries
Cop = CCPiProjectorSimple(ig, ag)

# Try forward and backprojection
b = Cop.direct(Phantom)
out2 = Cop.adjoint(b)

#%%
for i in range(b.get_dimension_size('vertical')):
    plt.imshow(b.subset(vertical=i).array)
    #plt.imshow(Phantom.subset( vertical=i).array)
    #plt.imshow(b.array[:,i,:])
    plt.show()
#%%

plt.imshow(out2.subset( vertical=0).array)
plt.show()

# Create least squares object instance with projector and data.
f = Norm2sq(Cop,b,c=0.5)

# Initial guess
x_init = ImageData(geometry=vg, dimension_labels=['horizontal_x','horizontal_y','vertical'])
#invL = 0.5
#g = f.grad(x_init)
#print (g)
#u = x_init - invL*f.grad(x_init)
        
#%%
# Run FISTA for least squares without regularization
opt = {'tol': 1e-4, 'iter': 100}
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt=opt)

plt.imshow(x_fista0.subset(vertical=0).array)
plt.title('FISTA0')
plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0,opt=opt)

plt.imshow(x_fista0.subset(vertical=0).array)
plt.title('FISTA1')
plt.show()

plt.semilogy(criter1)
plt.show()

# Run FBPD=Forward Backward Primal Dual method on least squares plus 1-norm
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)

plt.imshow(x_fbpd1.subset(vertical=0).array)
plt.title('FBPD1')
plt.show()

plt.semilogy(criter_fbpd1)
plt.show()

# Now FBPD for least squares plus TV
#lamtv = 1
#gtv = TV2D(lamtv)

#x_fbpdtv, it_fbpdtv, timing_fbpdtv, criter_fbpdtv = FBPD(x_init,None,f,gtv,opt=opt)

#plt.imshow(x_fbpdtv.subset(vertical=0).array)
#plt.show()

#plt.semilogy(criter_fbpdtv)
#plt.show()  


# Run CGLS, which should agree with the FISTA0
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Cop, b, opt=opt)

plt.imshow(x_CGLS.subset(vertical=0).array)
plt.title('CGLS')
plt.title('CGLS recon, compare FISTA0')
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()


#%%

clims = (0,1)
cols = 3
rows = 2
current = 1
fig = plt.figure()
# projections row
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(Phantom.as_array())))

imgplot = plt.imshow(Phantom.subset(vertical=0).as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA0')
imgplot = plt.imshow(x_fista0.subset(vertical=0).as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA1')
imgplot = plt.imshow(x_fista1.subset(vertical=0).as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD1')
imgplot = plt.imshow(x_fbpd1.subset(vertical=0).as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.subset(vertical=0).as_array(),vmin=clims[0],vmax=clims[1])

plt.show()
#%%
#current = current + 1
#a=fig.add_subplot(rows,cols,current)
#a.set_title('FBPD TV')
#imgplot = plt.imshow(x_fbpdtv.subset(vertical=0).as_array(),vmin=clims[0],vmax=clims[1])

fig = plt.figure()
# projections row
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter0 , label='FISTA0')
imgplot = plt.loglog(criter1 , label='FISTA1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD1')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
#imgplot = plt.loglog(criter_fbpdtv, label='FBPD TV')
b.legend(loc='right')
plt.show()
#%%