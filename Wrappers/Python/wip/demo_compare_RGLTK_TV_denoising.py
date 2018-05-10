
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, DataContainer
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, ZeroFun, Norm1, TV2D

from ccpi.optimisation.ops import LinearOperatorMatrix, Identity

from ccpi.plugins.regularisers import _ROF_TV_, _FGP_TV_

# Requires CVXPY, see http://www.cvxpy.org/
# CVXPY can be installed in anaconda using
# conda install -c cvxgrp cvxpy libgcc

# Whether to use or omit CVXPY
use_cvxpy = True
if use_cvxpy:
    from cvxpy import *

import numpy as np
import matplotlib.pyplot as plt


# Now try 1-norm and TV denoising with FBPD, first 1-norm.

# Set up phantom size NxN by creating ImageGeometry, initialising the 
# ImageData object with this geometry and empty array and finally put some
# data into its array, and display as image.
N = 64
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)
Phantom = ImageData(geometry=ig)

x = Phantom.as_array()
x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1

plt.imshow(x)
plt.title('Phantom image')
plt.show()

# Identity operator for denoising
I = Identity()

# Data and add noise
y = I.direct(Phantom)
np.random.seed(0)
y.array = y.array + 0.1*np.random.randn(N, N)

plt.imshow(y.array)
plt.title('Noisy image')
plt.show()

#%% TV parameter
lam_tv = 1.0

#%% Do CVX as high quality ground truth
if use_cvxpy:
    # Compare to CVXPY
    
    # Construct the problem.
    xtv_denoise = Variable(N,N)
    objectivetv_denoise = Minimize(0.5*sum_squares(xtv_denoise - y.array) + lam_tv*tv(xtv_denoise) )
    probtv_denoise = Problem(objectivetv_denoise)
    
    # The optimal objective is returned by prob.solve().
    resulttv_denoise = probtv_denoise.solve(verbose=False,solver=SCS,eps=1e-12)
    
    # The optimal solution for x is stored in x.value and optimal objective value 
    # is in result as well as in objective.value
    print("CVXPY least squares plus TV solution and objective value:")
    # print(xtv_denoise.value)
    # print(objectivetv_denoise.value)
    
plt.imshow(xtv_denoise.value)
plt.title('CVX TV')
plt.show()
print(objectivetv_denoise.value)


#%% THen FBPD

# Data fidelity term
f_denoise = Norm2sq(I,y,c=0.5)

# Initial guess
x_init_denoise = ImageData(np.zeros((N,N)))

gtv = TV2D(lam_tv)
gtv(gtv.op.direct(x_init_denoise))

opt_tv = {'tol': 1e-4, 'iter': 10000}

x_fbpdtv_denoise, itfbpdtv_denoise, timingfbpdtv_denoise, criterfbpdtv_denoise = FBPD(x_init_denoise, None, f_denoise, gtv,opt=opt_tv)


print("CVXPY least squares plus TV solution and objective value:")
plt.imshow(x_fbpdtv_denoise.as_array())
plt.title('FBPD TV')
plt.show()

print(criterfbpdtv_denoise[-1])

#%%
plt.loglog([0,opt_tv['iter']], [objectivetv_denoise.value,objectivetv_denoise.value], label='CVX TV')
plt.loglog(criterfbpdtv_denoise, label='FBPD TV')
plt.show()

#%% FISTA with ROF-TV regularisation
g_rof = _ROF_TV_(lambdaReg = lam_tv,iterationsTV=5000,tolerance=1e-5,time_marchstep=0.01,device='cpu')

xtv_rof = g_rof.prox(y,1.0)

print("CCPi-RGL TV ROF:")
plt.imshow(xtv_rof.as_array())
plt.title('ROF TV prox')
plt.show()
print(g_rof(xtv_rof))

#%% FISTA with FGP-TV regularisation
g_fgp = _FGP_TV_(lambdaReg = lam_tv,iterationsTV=5000,tolerance=1e-5,methodTV=0,nonnegativity=0,printing=0,device='cpu')

xtv_fgp = g_fgp.prox(y,1.0)

print("CCPi-RGL TV FGP:")
plt.imshow(xtv_fgp.as_array())
plt.title('FGP TV prox')
plt.show()
print(g_fgp(xtv_fgp))

# Compare all reconstruction
clims = (-0.2,1.2)
dlims = (-0.2,0.2)
cols = 3
rows = 2
current = 1

fig = plt.figure()
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD')
imgplot = plt.imshow(x_fbpdtv_denoise.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('ROF')
imgplot = plt.imshow(xtv_rof.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FGP')
imgplot = plt.imshow(xtv_fgp.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA LS+1')
imgplot = plt.imshow(x_fbpdtv_denoise.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD LS+1')
imgplot = plt.imshow(xtv_rof.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
plt.axis('off')

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD TV')
imgplot = plt.imshow(xtv_fgp.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
plt.axis('off')
