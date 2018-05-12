
# This demo illustrates how the CCPi Regularisation Toolkit can be used 
# as TV denoising for use with the FISTA algorithm of the modular 
# optimisation framework and compares with the FBPD TV implementation as well
# as CVXPY.

# All own imports
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, DataContainer
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, ZeroFun, Norm1, TV2D
from ccpi.optimisation.ops import LinearOperatorMatrix, Identity

from ccpi.plugins.regularisers import _ROF_TV_, _FGP_TV_, _SB_TV_

# All external imports
import numpy as np
import matplotlib.pyplot as plt

#%%
# Requires CVXPY, see http://www.cvxpy.org/
# CVXPY can be installed in anaconda using
# conda install -c cvxgrp cvxpy libgcc

# Whether to use or omit CVXPY
use_cvxpy = True
if use_cvxpy:
    from cvxpy import *

#%%

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

# Display noisy image
plt.imshow(y.array)
plt.title('Noisy image')
plt.show()

#%% TV parameter
lam_tv = 1.0

#%% Do CVX as high quality ground truth for comparison.
if use_cvxpy:
    
    # Construct the problem.
    xtv_denoise = Variable(N,N)
    objectivetv_denoise = Minimize(0.5*sum_squares(xtv_denoise - y.array) + lam_tv*tv(xtv_denoise) )
    probtv_denoise = Problem(objectivetv_denoise)
    
    # The optimal objective is returned by prob.solve().
    resulttv_denoise = probtv_denoise.solve(verbose=False,solver=SCS,eps=1e-12)
    
    # The optimal solution for x is stored in x.value and optimal objective 
    # value is in result as well as in objective.value
    
    # Display
    plt.figure()
    plt.imshow(xtv_denoise.value)
    plt.title('CVX TV  with objective equal to {:.2f}'.format(objectivetv_denoise.value))
    plt.show()
    print(objectivetv_denoise.value)

#%%
# Data fidelity term
f_denoise = Norm2sq(I,y,c=0.5)

#%%

#%% Then run FBPD algorithm for TV  denoising

# Initial guess
x_init_denoise = ImageData(np.zeros((N,N)))

# Set up TV function
gtv = TV2D(lam_tv)

# Evalutate TV of noisy image.
gtv(gtv.op.direct(y))

# Specify FBPD options and run FBPD.
opt_tv = {'tol': 1e-4, 'iter': 10000}
x_fbpdtv_denoise, itfbpdtv_denoise, timingfbpdtv_denoise, criterfbpdtv_denoise = FBPD(x_init_denoise, None, f_denoise, gtv,opt=opt_tv)

print("FBPD least squares plus TV solution and objective value:")
plt.figure()
plt.imshow(x_fbpdtv_denoise.as_array())
plt.title('FBPD TV with objective equal to {:.2f}'.format(criterfbpdtv_denoise[-1]))
plt.show()

print(criterfbpdtv_denoise[-1])

# Also plot history of criterion vs. CVX
if use_cvxpy:
    plt.loglog([0,opt_tv['iter']], [objectivetv_denoise.value,objectivetv_denoise.value], label='CVX TV')
plt.loglog(criterfbpdtv_denoise, label='FBPD TV')
plt.legend()
plt.show()

#%% FISTA with ROF-TV regularisation
g_rof = _ROF_TV_(lambdaReg = lam_tv,
                 iterationsTV=2000,
                 tolerance=0,
                 time_marchstep=0.0009,
                 device='cpu')

# Evaluating the proximal operator corresponds to denoising.
xtv_rof = g_rof.prox(y,1.0)

# Display denoised image and final criterion value.
print("CCPi-RGL TV ROF:")
plt.figure()
plt.imshow(xtv_rof.as_array())
EnergytotalROF = f_denoise(xtv_rof) + g_rof(xtv_rof)
plt.title('ROF TV prox with objective equal to {:.2f}'.format(EnergytotalROF))
plt.show()
print(EnergytotalROF)

#%% FISTA with FGP-TV regularisation
g_fgp = _FGP_TV_(lambdaReg = lam_tv,
                 iterationsTV=5000,
                 tolerance=0,
                 methodTV=0,
                 nonnegativity=0,
                 printing=0,
                 device='cpu')

# Evaluating the proximal operator corresponds to denoising.
xtv_fgp = g_fgp.prox(y,1.0)

# Display denoised image and final criterion value.
print("CCPi-RGL TV FGP:")
plt.figure()
plt.imshow(xtv_fgp.as_array())
EnergytotalFGP = f_denoise(xtv_fgp) + g_fgp(xtv_fgp)
plt.title('FGP TV prox with objective equal to {:.2f}'.format(EnergytotalFGP))
plt.show()
print(EnergytotalFGP)

#%% Split-Bregman-TV regularisation
g_sb = _SB_TV_(lambdaReg = lam_tv,
               iterationsTV=1000,
               tolerance=0,
               methodTV=0,
               printing=0,
               device='cpu')

# Evaluating the proximal operator corresponds to denoising.
xtv_sb = g_sb.prox(y,1.0)

# Display denoised image and final criterion value.
print("CCPi-RGL TV SB:")
plt.figure()
plt.imshow(xtv_sb.as_array())
EnergytotalSB = f_denoise(xtv_sb) + g_fgp(xtv_sb)
plt.title('SB TV prox with objective equal to {:.2f}'.format(EnergytotalSB))
plt.show()
print(EnergytotalSB)

#%%

# Compare all reconstruction
clims = (-0.2,1.2)
dlims = (-0.2,0.2)
cols = 4
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
a.set_title('SB')
imgplot = plt.imshow(xtv_sb.as_array(),vmin=clims[0],vmax=clims[1])
plt.axis('off')

if use_cvxpy: 
    current = current + 1
    a=fig.add_subplot(rows,cols,current)
    a.set_title('FBPD - CVX')
    imgplot = plt.imshow(x_fbpdtv_denoise.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
    plt.axis('off')
    
    current = current + 1
    a=fig.add_subplot(rows,cols,current)
    a.set_title('ROF - CVX')
    imgplot = plt.imshow(xtv_rof.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
    plt.axis('off')
    
    current = current + 1
    a=fig.add_subplot(rows,cols,current)
    a.set_title('FGP - CVX')
    imgplot = plt.imshow(xtv_fgp.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
    plt.axis('off')
    
    current = current + 1
    a=fig.add_subplot(rows,cols,current)
    a.set_title('SB - CVX')
    imgplot = plt.imshow(xtv_sb.as_array()-xtv_denoise.value,vmin=dlims[0],vmax=dlims[1])
    plt.axis('off')
