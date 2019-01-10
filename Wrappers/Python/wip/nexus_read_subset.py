# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 16:00:53 2018

@author: ofn77899
"""

import os
from ccpi.io.reader import NexusReader

from h5py import File as NexusFile
from sys import getsizeof
import matplotlib.pyplot as plt

import numpy as np

nf = NexusFile(os.path.join(".." ,".." ,".." , "..", 
                                  "CCPi-ReconstructionFramework","data" , "pco1-74240.hdf" ),
'r')

image_keys = np.array(nf['entry1/tomo_entry/instrument/detector/image_key'])


print (image_keys)

reader = NexusReader(os.path.join(".." ,".." ,".." , "..", 
                                  "CCPi-ReconstructionFramework","data" , 
                                  "pco1-74240.hdf"))

print (reader.get_sinogram_dimensions())

a = reader.load()



plt.imshow(a[:,100,:])

                
                
arr = np.array(nf['entry1/tomo_entry/data/data'][:,100,:][image_keys==0])

fig = plt.figure()

plt.imshow(arr)

fig = plt.figure()

b = reader.get_acquisition_data_subset(80,81)#.subset(['angle','horizontal'])
plt.imshow(b.array.squeeze())
fig = plt.figure()

b2 = reader.get_acquisition_data_subset(80,81).subset(['angle','horizontal'])
plt.imshow(b2.array)

fig = plt.figure()

b3 = reader.get_acquisition_data_slice(80)
plt.imshow(b3.array)