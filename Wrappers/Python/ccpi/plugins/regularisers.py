# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# This requires CCPi-Regularisation toolbox to be installed
from ccpi.filters import regularisers
from ccpi.filters.cpu_regularisers import TV_ENERGY
from ccpi.framework import DataContainer
from ccpi.optimisation.functions import Function
import numpy as np


class ROF_TV(Function):
    def __init__(self,lambdaReg,iterationsTV,tolerance,time_marchstep,device):
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.time_marchstep = time_marchstep
        self.device = device # string for 'cpu' or 'gpu'
    def __call__(self,x):
        # evaluate objective function of TV gradient
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    def prox(self,x,tau):
        pars = {'algorithm' : ROF_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'time_marching_parameter':self.time_marchstep}
        
        res , info = regularisers.ROF_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'], self.device)
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
        return out

class FGP_TV(Function):
    def __init__(self,lambdaReg,iterationsTV,tolerance,methodTV,nonnegativity,printing,device):
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
    def proximal(self,x,tau, out=None):
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
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
        return out

class FGP_dTV(Function):
    def __init__(self, refdata, regularisation_parameter, iterations,
                 tolerance, eta_const, methodTV, nonneg, device='cpu'):
        # set parameters
        self.lambdaReg = regularisation_parameter
        self.iterationsTV = iterations
        self.tolerance = tolerance
        self.methodTV = methodTV
        self.nonnegativity = nonneg
        self.device = device # string for 'cpu' or 'gpu'
        self.refData = np.asarray(refdata.as_array(), dtype=np.float32)
        self.eta = eta_const
        
    def __call__(self,x):
        # evaluate objective function of TV gradient
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    def proximal(self,x,tau, out=None):
        pars = {'algorithm' : FGP_dTV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV ,\
                'nonneg': self.nonnegativity ,\
                'eta_const' : self.eta,\
                'refdata':self.refData}
       #inputData, refdata, regularisation_parameter, iterations,
       #              tolerance_param, eta_const, methodTV, nonneg, device='cpu' 
        res , info = regularisers.FGP_dTV(pars['input'],
              pars['refdata'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'],
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],
              self.device)
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
        return out
        
class SB_TV(Function):
    def __init__(self,lambdaReg,iterationsTV,tolerance,methodTV,printing,device):
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
                'methodTV': self.methodTV ,\
                'printingOut': self.printing}
        
        res , info = regularisers.SB_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['printingOut'], self.device)
        if out is not None:
            out.fill(res)
        else:
            out = x.copy()
            out.fill(res)
        return out
