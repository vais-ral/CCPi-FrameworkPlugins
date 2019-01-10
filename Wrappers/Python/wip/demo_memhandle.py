# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 21:40:58 2018

@author: ofn77899
"""


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

b = Cop.direct(Phantom)

f = Norm2sq(Cop,b,c=0.5)
x_init = ImageData(geometry=ig,     
                   dimension_labels=['horizontal_x','horizontal_y','vertical'])
a = f.grad(x_init)
b = a.copy()
f.gradient(x_init, out=b)
print ("a-b" , (a-b).sum())
print ("f.L" , f.L)
u = x_init - f.grad(x_init)/f.L
x = g.prox(u,1/f.L)

um = x_init.clone()
xm = x_init.clone()
f.gradient(x_init, out=um)
um *= -1/f.L
um += x_init
#x = g.prox(u,invL)
g.proximal(um, 1/f.L, out=xm)
print ("x-xm" , (x-xm).sum())
          

