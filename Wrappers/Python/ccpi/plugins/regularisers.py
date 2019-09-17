# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ccpi.filters import regularisers
from ccpi.filters.cpu_regularisers import TV_ENERGY
from ccpi.optimisation.functions import Function
import numpy as np

class ROF_TV(Function):
    
    def __init__(self,lambdaReg,iterationsTV,tolerance,time_marchstep,device):
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.time_marchstep = time_marchstep
        self.device = device # string for 'cpu' or 'gpu'
        self.tolerance = tolerance
        
    def __call__(self,x):
        
        # evaluate objective function of TV gradient
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    
    def proximal(self, x, tau, out = None):
        
        pars = {'algorithm' : ROF_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'time_marching_parameter':self.time_marchstep,\
                'tolerance':self.tolerance}
        
        res , info = regularisers.ROF_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'], pars['tolerance'], self.device)
        
        self.info = info
        
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
            return out
    
class FGP_TV(Function):
    
    def __init__(self, lambdaReg, iterationsTV, tolerance, methodTV, nonnegativity, printing, device):
        
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.tolerance = tolerance
        self.methodTV = methodTV
        self.nonnegativity = nonnegativity
        self.printing = printing
        self.device = device # string for 'cpu' or 'gpu'
        
    def __call__(self,x):
        # evaluate objective function of TV gradient
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    
    def proximal(self, x, tau, out = None):
        
        pars = {'algorithm' : FGP_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV ,\
                'nonneg': self.nonnegativity ,\
                'printingOut': self.printing}
        
        res , info = regularisers.FGP_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],
              self.device)
        
        self.info = info
                
        if out is not None:
            out.fill(res)
        else:
            tmp = x.copy()
            tmp.fill(res)
            return tmp
                
    def convex_conjugate(self,x):        
        return 0.0
    
    
class TGV(Function):

    def __init__(self, regularisation_parameter, alpha1, alpha2, iter_TGV, LipshitzConstant, torelance, device ):
        
        self.regularisation_parameter = regularisation_parameter
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.iter_TGV = iter_TGV
        self.LipshitzConstant = LipshitzConstant
        self.torelance = torelance
        self.device = device
        
    
    def __call__(self,x):
        
        # TODO this is not correct, need a TGV energy same as TV
        return 0.0
    
    def proximal(self, x, tau, out=None):
        
        pars = {'algorithm' : TGV, \
                'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularisation_parameter':self.regularisation_parameter, \
                'alpha1':self.alpha1,\
                'alpha0':self.alpha2,\
                'number_of_iterations' :self.iter_TGV ,\
                'LipshitzConstant' :self.LipshitzConstant ,\
                'tolerance_constant':self.torelance}
        
        res , info = regularisers.TGV(pars['input'], 
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['tolerance_constant'],self.device)
                
        # info: return number of iteration and reached tolerance
        # https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/master/src/Core/regularisers_CPU/TGV_core.c#L168
        # Stopping Criteria  || u^k - u^(k-1) ||_{2} / || u^{k} ||_{2}    
  
        self.info = info
        
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
            return out        
    
    def convex_conjugate(self, x):
        
        # TODO this is not correct
        return 0.0

        
class LLT_ROF(Function):
    
    
    def __init__(self, regularisation_parameterROF, 
                       regularisation_parameterLLT,
                       iter_LLT_ROF, time_marching_parameter, torelance, device ):
        
        self.regularisation_parameterROF = regularisation_parameterROF
        self.regularisation_parameterLLT = regularisation_parameterLLT
        self.iter_LLT_ROF = iter_LLT_ROF
        self.time_marching_parameter = time_marching_parameter
        self.torelance = torelance
        self.device = device 
        
    def __call__(self,x):
        
        # TODO this is not correct
        return 0.0        
    
    def proximal(self, x, tau, out=None):
        
        pars = {'algorithm' : LLT_ROF, \
                'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularisation_parameterROF':self.regularisation_parameterROF, \
                'regularisation_parameterLLT':self.regularisation_parameterLLT,
                'number_of_iterations' :self.iter_LLT_ROF ,\
                'time_marching_parameter': self.time_marching_parameter,\
                'tolerance_constant':self.torelance}
        
        
        
        res , info = regularisers.LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'],self.device)
                 
        # info: return number of iteration and reached tolerance
        # https://github.com/vais-ral/CCPi-Regularisation-Toolkit/blob/master/src/Core/regularisers_CPU/TGV_core.c#L168
        # Stopping Criteria  || u^k - u^(k-1) ||_{2} / || u^{k} ||_{2}    
  
        self.info = info
        
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
            return out        
    
    def convex_conjugate(self, x):
        
        # TODO this is not correct
        return 0.0    
    
       
class SB_TV(Function):
    
    def __init__(self, lambdaReg, iterationsTV, tolerance, methodTV, printing, device):
        
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.tolerance = tolerance
        self.methodTV = methodTV
        self.printing = printing
        self.device = device # string for 'cpu' or 'gpu'
        
    def __call__(self,x):
        
        # evaluate objective function of TV gradient
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    
    def proximal(self,x,tau, out=None):
        pars = {'algorithm' : SB_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV}
        
        res , info = regularisers.SB_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'], self.device)
        
        self.info = info
    
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
            return out


class FGP_dTV(Function):
    
    def __init__(self,regularisation_parameter,iterationsdTV,tolerance,
                 eta_const, methodTV,nonneg,device):
        
        # set parameters
        self.regularisation_parameter = regularisation_parameter
        self.iterationsdTV = iterationsdTV
        self.tolerance = tolerance
        self.methodTV = methodTV
        self.eta_const = eta_const
        self.nonneg = nonneg
        self.device = device # string for 'cpu' or 'gpu'
        
    def __call__(self,x):
        
        # TODO this is not correct
        return 0.0
    
    def proximal(self, x, tau, out=None):
        
        pars = {'algorithm' : FGP_dTV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
               'ref_data' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.regularisation_parameter, \
                'number_of_iterations' :self.iterationsdTV ,\
                'tolerance_constant':self.tolerance,\
                'eta_const':self.eta_const,\
                'nonneg':self.nonneg,\
                'methodTV': self.methodTV}
        
        res , info = regularisers.FGP_dTV(pars['input'], 
              pars['ref_data'],
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'], 
              pars['nonneg'], 
              self.device)
        
        self.info = info
    
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
            return out

class TNV(Function):
    
    def __init__(self,regularisation_parameter,iterationsTNV,tolerance):
        
        # set parameters
        self.regularisation_parameter = regularisation_parameter
        self.iterationsTNV = iterationsTNV
        self.tolerance = tolerance

        
    def __call__(self,x):
        
        #TODO this is not correct
        return 0.0
    
    def proximal(self, x, tau, out = None):
        
        pars = {'algorithm' : TNV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularisation_parameter':self.regularisation_parameter, \
                'number_of_iterations' :self.iterationsTNV,\
                'tolerance_constant':self.tolerance}
        
        res, info   = regularisers.TNV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'])
        
        self.info = info
    
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
            return out
