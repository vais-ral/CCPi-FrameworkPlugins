#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of CPU regularisers 

@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, TNV, NDF, Diff4th
from ccpi.filters.regularisers import PatchSelect, NLTV
from ccpi.supp.qualitymetrics import QualityTools

from ccpi.plugins.regularisers import TNV as TNV_new
###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            elif key == 'refdata':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################

filename = 'lena_gray_512.tif'

# read image
Im = plt.imread(filename)
Im = np.asarray(Im, dtype='float32')

Im = Im/255.0
perc = 0.05
u0 = Im + np.random.normal(loc = 0 ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
u_ref = Im + np.random.normal(loc = 0 ,
                                  scale = 0.01 * Im , 
                                  size = np.shape(Im))
(N,M) = np.shape(u0)
# map the u0 u0->u0>0
# f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')

# change dims to check that modules work with non-squared images
"""
M = M-100
u_ref2 = np.zeros([N,M],dtype='float32')
u_ref2[:,0:M] = u_ref[:,0:M]
u_ref = u_ref2
del u_ref2

u02 = np.zeros([N,M],dtype='float32')
u02[:,0:M] = u0[:,0:M]
u0 = u02
del u02

Im2 = np.zeros([N,M],dtype='float32')
Im2[:,0:M] = Im[:,0:M]
Im = Im2
del Im2
"""

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("__________Total nuclear Variation__________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of TNV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

channelsNo = 5
noisyVol = np.zeros((channelsNo,N,M),dtype='float32')
idealVol = np.zeros((channelsNo,N,M),dtype='float32')

for i in range (channelsNo):
    noisyVol[i,:,:] = Im + np.random.normal(loc = 0 , scale = perc * Im , size = np.shape(Im))
    idealVol[i,:,:] = Im

# set parameters
pars = {'algorithm' : TNV, \
        'input' : noisyVol,\
        'regularisation_parameter': 0.04, \
        'number_of_iterations' : 200 ,\
        'tolerance_constant':1e-05
        }
        
print ("#############TNV CPU#################")
start_time = timeit.default_timer()
tnv_cpu = TNV(pars['input'],           
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'])

Qtools = QualityTools(idealVol, tnv_cpu)
pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(tnv_cpu[3,:,:], cmap="gray")
plt.title('{}'.format('CPU results'))

#%%
from ccpi.framework import ImageData
g = TNV_new(0.04, 200, 1e-5)
sol = g.proximal(ImageData(noisyVol), 1)

#%%
plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
plt.imshow(sol.as_array()[2])
plt.colorbar()

plt.subplot(3,1,2)
plt.imshow(tnv_cpu[2])
plt.colorbar()

plt.subplot(3,1,3)
plt.imshow(np.abs(tnv_cpu[2] - sol.as_array()[2]))
plt.colorbar()

plt.show()

#plt.imshow(sol.as_array())