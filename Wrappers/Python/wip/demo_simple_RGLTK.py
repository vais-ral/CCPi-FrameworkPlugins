
# This demo illustrates how the CCPi Regularisation Toolkit can be used 
# as TV regularisation for use with the FISTA algorithm of the modular 
# optimisation framework and compares with the FBPD TV implementation.

# All own imports
from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D
from ccpi.astra.ops import AstraProjectorSimple
from ccpi.plugins.regularisers import ROF_TV, FGP_TV, SB_TV

# All external imports
import numpy as np
import matplotlib.pyplot as plt

# Choose either a parallel-beam (1=parallel2D) or fan-beam (2=cone2D) test case
test_case = 1

# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 128
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
plt.show()

# Set up AcquisitionGeometry object to hold the parameters of the measurement
# setup geometry: # Number of angles, the actual angles from 0 to 
# pi for parallel beam and 0 to 2pi for fanbeam, set the width of a detector 
# pixel relative to an object pixel, the number of detector pixels, and the 
# source-origin and origin-detector distance (here the origin-detector distance 
# set to 0 to simulate a "virtual detector" with same detector pixel size as
# object pixel size).
angles_num = 20
det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

if test_case==1:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             det_num,det_w)
elif test_case==2:
    angles = np.linspace(0,2*np.pi,angles_num,endpoint=False)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             det_num,
                             det_w,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)
else:
    NotImplemented

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to ASTRA as well as specifying whether to use CPU or GPU.
Aop = AstraProjectorSimple(ig, ag, 'cpu')

# Forward and backprojection are available as methods direct and adjoint. Here 
# generate test data b and do simple backprojection to obtain z.
b = Aop.direct(Phantom)
z = Aop.adjoint(b)

plt.imshow(b.array)
plt.title('Simulated data')
plt.show()

plt.imshow(z.array)
plt.title('Backprojected data')
plt.show()

# Create least squares object instance with projector and data.
f = Norm2sq(Aop,b,c=0.5)

# Initial guess
x_init = ImageData(np.zeros(x.shape),geometry=ig)

# Set up FBPD algorithm for TV reconstruction and solve
opt_FBPD = {'tol': 1e-4, 'iter': 20000}

lamtv = 1.0
gtv = TV2D(lamtv)

x_fbpdtv, it_fbpdtv, timing_fbpdtv, criter_fbpdtv = FBPD(x_init,
                                                         None,
                                                         f,
                                                         gtv,
                                                         opt=opt_FBPD)

plt.figure()
plt.subplot(121)
plt.imshow(x_fbpdtv.array)
plt.title('FBPD TV')
plt.subplot(122)
plt.semilogy(criter_fbpdtv)
plt.show()

# Set up the ROF variant of TV from the CCPi Regularisation Toolkit and run
# TV-reconstruction using FISTA
g_rof = ROF_TV(lambdaReg = lamtv,
                 iterationsTV=50,
                 tolerance=1e-5,
                 time_marchstep=0.01,
                 device='cpu')

opt = {'tol': 1e-4, 'iter': 100}

x_fista_rof, it1, timing1, criter_rof = FISTA(x_init, f, g_rof,opt)

plt.figure()
plt.subplot(121)
plt.imshow(x_fista_rof.array)
plt.title('FISTA ROF TV')
plt.subplot(122)
plt.semilogy(criter_rof)
plt.show()

# Repeat for FGP variant.
g_fgp = FGP_TV(lambdaReg = lamtv,
                 iterationsTV=50,
                 tolerance=1e-5,
                 methodTV=0,
                 nonnegativity=0,
                 printing=0,
                 device='cpu')

x_fista_fgp, it1, timing1, criter_fgp = FISTA(x_init, f, g_fgp,opt)

plt.figure()
plt.subplot(121)
plt.imshow(x_fista_fgp.array)
plt.title('FISTA FGP TV')
plt.subplot(122)
plt.semilogy(criter_fgp)
plt.show()

# Repeat for SB variant.
g_sb = SB_TV(lambdaReg = lamtv,
                 iterationsTV=50,
                 tolerance=1e-5,
                 methodTV=0,
                 printing=0,
                 device='cpu')

x_fista_sb, it1, timing1, criter_sb = FISTA(x_init, f, g_sb,opt)

plt.figure()
plt.subplot(121)
plt.imshow(x_fista_sb.array)
plt.title('FISTA SB TV')
plt.subplot(122)
plt.semilogy(criter_sb)
plt.show()

# Compare all reconstruction and criteria
clims = (0,1)
cols = 4
rows = 1
current = 1
fig = plt.figure()

a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD TV')
imgplot = plt.imshow(x_fbpdtv.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA ROF TV')
imgplot = plt.imshow(x_fista_rof.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA FGP TV')
imgplot = plt.imshow(x_fista_fgp.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA SB TV')
imgplot = plt.imshow(x_fista_sb.as_array(),vmin=clims[0],vmax=clims[1])

fig = plt.figure()

b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter_fbpdtv , label='FBPD TV')
imgplot = plt.loglog(criter_rof , label='ROF TV')
imgplot = plt.loglog(criter_fgp, label='FGP TV')
imgplot = plt.loglog(criter_sb, label='SB TV')
b.legend(loc='right')
plt.show()
