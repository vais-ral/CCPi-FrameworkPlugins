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
from ccpi.optimisation.funcs import Function
import numpy as np

class Regularisers(Function):
    def __init__(self, **kwargs):
        super(Regulariser, self).__init__(**kwargs)
        
    def __call__(self,x):
        '''evaluate objective function of TV gradient'''
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        EnergyValTV = TV_ENERGY(x32 , x32 , self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    def get_instance(self, regulariser, **kwargs):
        kwargs.setdefault('lambdaReg', 1e-5)
        kwargs.setdefault('max_iterations', 100)
        kwargs.setdefault('tolerance', 1e-5)
        kwargs.setdefault('device', 'cpu')
        
        if regulariser.__name__ == 'ROF_TV':
            required_parameters = ['time_marchstep']

        elif regulariser.__name__ == 'FGP_TV':
            required_parameters = ['nonneg' , 'methodTV' , 'printingOut']
        
        # check we have all parameters
        missing = []
        for el in required_parameters:
            if el not in self.kwargs.keys():
                missing.append(el)
                
        if len(missing) > 0:
            raise ValueError('Missing parameters: {}'.format(missing))        
        #
        
        def proximal(x, tau, out=None):
            if not x.dtype == np.float32:
                x32 = np.asarray(x.as_array(), dtype=np.float32)
            else:
                x32 = x
            return regulariser(x32,tau*kwargs['lambdaReg'], **kwargs)
        self.proximal = proximal
            
        return self

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
        out = regularisers.ROF_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'], self.device)

        return type(x)(out, geometry=x.geometry,
                       dimension_labels=x.dimension_labels)
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
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        EnergyValTV = TV_ENERGY(x32 , x32 , self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    def prox(self,x,tau):
        pars = {'algorithm' : FGP_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV ,\
                'nonneg': self.nonnegativity ,\
                'printingOut': self.printing}
        out = regularisers.FGP_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'], self.device)
        return type(x)(out, geometry=x.geometry,
                       dimension_labels=x.dimension_labels)
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
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        EnergyValTV = TV_ENERGY(x32 , x32 , self.lambdaReg, 2)
        return 0.5*EnergyValTV[0]
    def prox(self,x,tau):
        pars = {'algorithm' : SB_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*tau, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV ,\
                'printingOut': self.printing}
        out = regularisers.SB_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['printingOut'], self.device)
        return type(x)(out, geometry=x.geometry,
                       dimension_labels=x.dimension_labels)
class NDF(Function):
    '''A Function wrapper for the NDF regulariser

    NDF(inputData, regularisation_parameter, edge_parameter, iterations,
                     time_marching_parameter, penalty_type, device='cpu')
    '''
    def __init__(self, regularisation_parameter, edge_parameter, iterations,
                     time_marching_parameter, penalty_type, device='cpu'):
        # set parameters
        self.regularisation_parameter = regularisation_parameter
        self.edge_parameter = edge_parameter
        self.iterations = iterations
        self.time_marching_parameter = time_marching_parameter
        self.penalty_type = penalty_type
        self.device = device # string for 'cpu' or 'gpu'
    def __call__(self,x):
        # evaluate objective function of TV gradient
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        EnergyValTV = TV_ENERGY(x32 , x32 , self.regularisation_parameter, 2)
        return 0.5*EnergyValTV[0]
    def prox(self,x,tau):
        return self.proximal (x,tau,out=None)
    def proximal(self, x, tau, out=None):
        if out is not None:
            raise ValueError('out cannot be passed as argument yet')
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        pars = {'algorithm' : NDF, \
        'input' : x32,\
        'regularisation_parameter':self.regularisation_parameter, \
        'edge_parameter':self.edge_parameter,\
        'number_of_iterations' :self.iterations ,\
        'time_marching_parameter':self.time_marching_parameter,\
        'penalty_type':  self.penalty_type,\
        'device' : self.device
        }
        out = NDF(pars['input'],
              pars['regularisation_parameter'],
              pars['edge_parameter'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['penalty_type'],
              pars['device'])
       return type(x)(out, geometry=x.geometry,
                       dimension_labels=x.dimension_labels)
class TGV(Function):
    '''A Function wrapper for the TGV regulariser

    TGV(inputData, regularisation_parameter, alpha1, alpha0, iterations,
                     LipshitzConst, device='cpu'):
    '''
    def __init__(self, regularisation_parameter, alpha1, alpha0, iterations,
                     LipshitzConst, device='cpu'):
        # set parameters
        self.regularisation_parameter = regularisation_parameter
        self.alpha1 = alpha1
        self.alpha0 = alpha0
        self.iterations = iterations
        self.LipshitzConst = LipshitzConst
        self.device = device # string for 'cpu' or 'gpu'
    def __call__(self,x):
        # evaluate objective function of TV gradient
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        EnergyValTV = TV_ENERGY(x32 , x32 , self.regularisation_parameter, 2)
        return 0.5*EnergyValTV[0]
    def prox(self,x,tau):
        return self.proximal (x,tau,out=None)
    def proximal(self, x, tau, out=None):
        if out is not None:
            raise ValueError('out cannot be passed as argument yet')
        if x.dtype != np.float32:
            x32 = np.asarray(x.as_array(), dtype=np.float32)
        else:
            x32 = x
        pars = {'algorithm' : TGV, \
        'input' : x32,\
        'regularisation_parameter':self.regularisation_parameter, \
        'alpha1':self.alpha1,\
        'alpha0':self.alpha0,\
        'number_of_iterations' :self.iterations ,\
        'LipshitzConstant' :self.LipshitzConstant,\
        'device' : self.device
        }
        out = TGV(pars['input'],
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['device'])
       return type(x)(out, geometry=x.geometry,
                      dimension_labels=x.dimension_labels)