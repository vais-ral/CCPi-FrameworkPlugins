
# This demo illustrates how CCPi 2D parallel-beam projectors can be used with
# the modular optimisation framework. The demo sets up a small 4-slice 3D test 
# case and demonstrates reconstruction using CGLS, as well as FISTA for least 
# squares and 1-norm regularisation and FBPD for 1-norm regularisation.

# First make all imports
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry

from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1

from ccpi.plugins.ops import CCPiProjectorSimple

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
angles_num = 20
det_w = 1.0
det_num = N

angles = np.linspace(0,np.pi,angles_num,endpoint=False,dtype=np.float32)*\
             180/np.pi

# Inputs: Geometry, 2D or 3D, angles, horz detector pixel count, 
#         horz detector pixel size, vert detector pixel count, 
#         vert detector pixel size.
ag = AcquisitionGeometry('parallel',
                         '3D',
                         angles,
                         N, 
                         det_w,
                         vert,
                         det_w)

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to CCPi projector.
Cop = CCPiProjectorSimple(ig, ag)

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z. Display all
#  data slices as images, and a single backprojected slice.
b = Cop.direct(Phantom)
z = Cop.adjoint(b)

for i in range(b.get_dimension_size('vertical')):
    plt.imshow(b.subset(vertical=i).array)
    plt.show()

plt.imshow(z.subset(vertical=0).array)
plt.title('Backprojected data')
plt.show()

# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial 
# guess and some algorithm options to be set. Note that 100 iterations for 
# some of the methods is a very low number and 1000 or 10000 iterations may be
# needed if one wants to obtain a converged solution.
x_init = ImageData(geometry=ig, 
                   dimension_labels=['horizontal_x','horizontal_y','vertical'])
opt = {'tol': 1e-4, 'iter': 100}

# First a CGLS reconstruction can be done:
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Cop, b, opt=opt)

plt.imshow(x_CGLS.subset(vertical=0).array)
plt.title('CGLS')
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()

# CGLS solves the simple least-squares problem. The same problem can be solved 
# by FISTA by setting up explicitly a least squares function object and using 
# no regularisation:

# Create least squares object instance with projector, test data and a constant 
# coefficient of 0.5:
f = Norm2sq(Cop,b,c=0.5)
opt['memopts'] = False
opt['log'] = True
# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt=opt)
print ("FISTA memhandle False: ", timing0[-1], criter0[-1])
opt['memopts'] = True
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt=opt)
print ("FISTA memhandle True: ", timing0[-1], criter0[-1])

plt.imshow(x_fista0.subset(vertical=0).array)
plt.title('FISTA Least squares')
plt.show()

plt.semilogy(criter0)
plt.title('FISTA Least squares criterion')
plt.show()

# FISTA can also solve regularised forms by specifying a second function object
# such as 1-norm regularisation with choice of regularisation parameter lam:

# Create 1-norm function object
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)

plt.imshow(x_fista1.subset(vertical=0).array)
plt.title('FISTA Least squares plus 1-norm regularisation')
plt.show()

plt.semilogy(criter1)
plt.title('FISTA Least squares plus 1-norm regularisation criterion')
plt.show()

# The least squares plus 1-norm regularisation problem can also be solved by 
# other algorithms such as the Forward Backward Primal Dual algorithm. This
# algorithm minimises the sum of three functions and the least squares and 
# 1-norm functions should be given as the second and third function inputs. 
# In this test case, this algorithm requires more iterations to converge, so
# new options are specified.
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)

plt.imshow(x_fbpd1.subset(vertical=0).array)
plt.title('FBPD for least squares plus 1-norm regularisation')
plt.show()

plt.semilogy(criter_fbpd1)
plt.title('FBPD for least squares plus 1-norm regularisation criterion')
plt.show()


# Compare all reconstruction and criteria

clims = (0,1)
cols = 3
rows = 2
current = 1

fig = plt.figure()
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(Phantom.as_array())))
imgplot = plt.imshow(Phantom.subset(vertical=0).as_array(),
                     vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.subset(vertical=0).as_array(),
                     vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS')
imgplot = plt.imshow(x_fista0.subset(vertical=0).as_array(),
                     vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS+1')
imgplot = plt.imshow(x_fista1.subset(vertical=0).as_array(),
                     vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD LS+1')
imgplot = plt.imshow(x_fbpd1.subset(vertical=0).as_array(),
                     vmin=clims[0],vmax=clims[1])
plt.axis('off')

fig = plt.figure()
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
imgplot = plt.loglog(criter0 , label='FISTA LS')
imgplot = plt.loglog(criter1 , label='FISTA LS+1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD LS+1')
b.legend(loc='lower left')
plt.show()