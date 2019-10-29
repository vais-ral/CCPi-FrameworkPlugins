from ccpi.framework import ImageData, TestData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, \
                        Gradient, SymmetrizedGradient, ZeroOperator
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, L2NormSquared
import os
import sys
from ccpi.plugins.regularisers import TGV, LLT_ROF

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
print ("Applying {} noise")

if len(sys.argv) > 2:
    method = sys.argv[2]
else:
    method = '0'
print ("method ", method)


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SHAPES)
ig = data.geometry
ag = ig

# Create noisy data. 
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2, seed=10)
elif noise == 'poisson':
    scale = 5
    n1 = TestData.random_noise(data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ImageData(n1)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Regularisation Parameter depending on the noise distribution
if noise == 's&p':
    alpha = 0.8
elif noise == 'poisson':
    alpha = .3
elif noise == 'gaussian':
    alpha = .2

beta = 2 * alpha

# Fidelity
if noise == 's&p':
    f3 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f3 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f3 = 0.5 * L2NormSquared(b=noisy_data)

if method == '0':
    
    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())
        
    op31 = Identity(ig, ag)
    op32 = ZeroOperator(op22.domain_geometry(), ag)
    
    operator = BlockOperator(op11, -1*op12, op21, op22, op31, op32, shape=(3,2) ) 
        
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm() 
    
    f = BlockFunction(f1, f2, f3)         
    g = ZeroFunction()
        
else:
    
    # Create operators
    op11 = Gradient(ig)
    op12 = Identity(op11.range_geometry())
    op22 = SymmetrizedGradient(op11.domain_geometry())    
    op21 = ZeroOperator(ig, op22.range_geometry())    
    
    operator = BlockOperator(op11, -1*op12, op21, op22, shape=(2,2) )      
    
    f1 = alpha * MixedL21Norm()
    f2 = beta * MixedL21Norm()     
    
    f = BlockFunction(f1, f2)         
    g = BlockFunction(f3, ZeroFunction())
     
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
#sigma = 1/normK
#tau = 1/normK

sigma = 1
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 5000
pdhg.update_objective_interval = 500
pdhg.run(5000)

# Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,4,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,4,3)
plt.imshow(pdhg.get_output()[0].as_array())
plt.title('TGV Reconstruction')
plt.colorbar()
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output()[0].as_array()[int(ig.shape[0]/2),:], label = 'TGV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()

#%%  Run CCPi-regulariser
# The TGV implementation is using PDHG algorithm with fixed 
# sigma = tau = 1/sqrt(12)

# There is an early stopping criteria 
# https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/master/src/Core/regularisers_CPU/TGV_core.c#L168

g = TGV(1, alpha, beta, 2000, normK**2, 1e-6, 'gpu')
#alphaROF = 0.1
#alphaLLT = 0.05
#g = LLT_ROF(alphaROF, alphaLLT, 500, 0.001, 1e-6, 'gpu')
sol = g.proximal(noisy_data, 1)

plt.imshow(sol.as_array())
plt.show()

plt.imshow(pdhg.get_output()[0].as_array())
plt.show()

plt.imshow(np.abs(sol.as_array() - pdhg.get_output()[0].as_array()))
plt.colorbar()
plt.show()

#%%

plt.plot(np.linspace(0,299,300),sol.as_array()[100,:], np.linspace(0,299,300),pdhg.get_output()[0].as_array()[100,:])

