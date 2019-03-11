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

import numpy
from ccpi.optimisation.ops import Operator, PowerMethodNonsquare
from ccpi.framework import ImageData, DataContainer , \
                           ImageGeometry, AcquisitionGeometry
from ccpi.plugins.processors import CCPiBackwardProjector, \
                                    CCPiForwardProjector , setupCCPiGeometries
        


class CCPiProjectorSimple(Operator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp, default=False):
        super(CCPiProjectorSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.acquisition_geometry = geomp
        self.volume_geometry = geomv
        
        if geomp.geom_type == "cone":
            raise TypeError('Can only handle parallel beam')
        
        # set-up the geometries if compatible
        geoms = setupCCPiGeometries(geomv.voxel_num_x,geomv.voxel_num_y, 
                                    geomv.voxel_num_z, geomp.angles, 0)
        

        vg = ImageGeometry(voxel_num_x=geoms['output_volume_x'],
                           voxel_num_y=geoms['output_volume_y'], 
                           voxel_num_z=geoms['output_volume_z'])

        pg = AcquisitionGeometry('parallel',
                          '3D',
                          geomp.angles,
                          geoms['n_h'], geomp.pixel_size_h,
                          geoms['n_v'], geomp.pixel_size_v #2D in 3D is a slice 1 pixel thick
                          )
        if not default:
            # check if geometry is the same (on the voxels)
            if not ( vg.voxel_num_x == geomv.voxel_num_x and \
                     vg.voxel_num_y == geomv.voxel_num_y and \
                     vg.voxel_num_z == geomv.voxel_num_z ):
                msg = 'The required volume geometry will not work\nThe following would\n'
                msg += vg.__str__()
                raise ValueError(msg)
            if not (pg.pixel_num_h == geomp.pixel_num_h and \
                    pg.pixel_num_v == geomp.pixel_num_v and \
                    len( pg.angles ) == len( geomp.angles ) ) :
                msg = 'The required acquisition geometry will not work\nThe following would\n'
                msg += pg.__str__()
                raise ValueError(msg)
        
        self.fp = CCPiForwardProjector(image_geometry=vg,
                                       acquisition_geometry=pg,
                                       output_axes_order=['angle','vertical','horizontal'])
        
        self.bp = CCPiBackwardProjector(image_geometry=vg,
                                    acquisition_geometry=pg,
                                    output_axes_order=['horizontal_x','horizontal_y','vertical'])
                
        # Initialise empty for singular value.
        self.s1 = None
        self.ag = pg
        self.vg = vg

    def is_linear(self):
        return True

    def direct(self, image_data, out=None):
        self.fp.set_input(image_data)
        if out is None:
            out = self.fp.get_output()
            return out
        else:
            out.fill(self.fp.get_output())

    def adjoint(self, acquisition_data, out=None):
        self.bp.set_input(acquisition_data)
        if out is None:
            out = self.bp.get_output()
            return out
        else:
            out.fill(self.bp.get_output())
    
    #def delete(self):
    #    astra.data2d.delete(self.proj_id)
    
    def get_max_sing_val(self):
        a = PowerMethodNonsquare(self,10)
        self.s1 = a[0] 
        return self.s1
    
    def size(self):
        # Only implemented for 3D
        return ( (self.acquisition_geometry.angles.size, \
                  self.acquisition_geometry.pixel_num_v,
                  self.acquisition_geometry.pixel_num_h), \
                 (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y,
                  self.volume_geometry.voxel_num_z) )
    def create_image_data(self):
        x0 = ImageData(geometry = self.volume_geometry, 
                       dimension_labels=self.bp.output_axes_order)#\
                       #.subset(['horizontal_x','horizontal_y','vertical'])
        x0.fill(numpy.random.randn(*x0.shape))
        return x0
    def domain_geometry(self):
        return ImageGeometry(
            self.vg.voxel_num_x,
            self.vg.voxel_num_y,
            self.vg.voxel_num_z,
            self.vg.voxel_size_x,
            self.vg.voxel_size_y,
            self.vg.voxel_size_z,
            self.vg.center_x,
            self.vg.center_y,
            self.vg.center_z,
            self.vg.channels,
            ['horizontal_x','horizontal_y','vertical'] )

    def range_geometry(self):
        return AcquisitionGeometry(self.ag.geom_type,
                               self.ag.dimension,
                               self.ag.angles,
                               self.ag.pixel_num_h,
                               self.ag.pixel_size_h,
                               self.ag.pixel_num_v,
                               self.ag.pixel_size_v,
                               self.ag.dist_source_center,
                               self.ag.dist_center_detector,
                               self.ag.channels,
                               ['angle','vertical','horizontal'])

