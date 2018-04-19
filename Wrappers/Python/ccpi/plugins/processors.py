# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License

from ccpi.framework import DataProcessor, DataContainer, AcquisitionData,\
 AcquisitionGeometry, ImageGeometry, ImageData
from ccpi.reconstruction.parallelbeam import alg as pbalg
import numpy
import h5py
from scipy import ndimage

import matplotlib.pyplot as plt


def setupCCPiGeometries(voxel_num_x, voxel_num_y, voxel_num_z, angles, counter):
    
    vg = ImageGeometry(voxel_num_x=voxel_num_x,voxel_num_y=voxel_num_y, voxel_num_z=voxel_num_z)
    Phantom_ccpi = ImageData(geometry=vg,
                        dimension_labels=['horizontal_x','horizontal_y','vertical'])
    #.subset(['horizontal_x','horizontal_y','vertical'])
    # ask the ccpi code what dimensions it would like
        
    voxel_per_pixel = 1
    geoms = pbalg.pb_setup_geometry_from_image(Phantom_ccpi.as_array(),
                                                angles,
                                                voxel_per_pixel )
    
    pg = AcquisitionGeometry('parallel',
                              '3D',
                              angles,
                              geoms['n_h'], 1.0,
                              geoms['n_v'], 1.0 #2D in 3D is a slice 1 pixel thick
                              )
    
    center_of_rotation = Phantom_ccpi.get_dimension_size('horizontal_x') / 2
    ad = AcquisitionData(geometry=pg,dimension_labels=['angle','vertical','horizontal'])
    geoms_i = pbalg.pb_setup_geometry_from_acquisition(ad.as_array(),
                                                angles,
                                                center_of_rotation,
                                                voxel_per_pixel )

    #print (counter)
    counter+=1
    #print (geoms , geoms_i)
    if counter < 4:
        if (not ( geoms_i == geoms )):
            print ("not equal and {0}".format(counter))
            X = max(geoms['output_volume_x'], geoms_i['output_volume_x'])
            Y = max(geoms['output_volume_y'], geoms_i['output_volume_y'])
            Z = max(geoms['output_volume_z'], geoms_i['output_volume_z'])
            return setupCCPiGeometries(X,Y,Z,angles, counter)
        else:
            print ("return geoms {0}".format(geoms))
            return geoms
    else:
        print ("return geoms_i {0}".format(geoms_i))
        return geoms_i


class CCPiForwardProjector(DataProcessor):
    '''Normalization based on flat and dark
    
    This processor read in a AcquisitionData and normalises it based on 
    the instrument reading with and without incident photons or neutrons.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self,
                 image_geometry       = None, 
                 acquisition_geometry = None,
                 output_axes_order    = None):
        if output_axes_order is None:
            # default ccpi projector image storing order
            output_axes_order = ['angle','vertical','horizontal']
        
        kwargs = {
                  'image_geometry'       : image_geometry, 
                  'acquisition_geometry' : acquisition_geometry,
                  'output_axes_order'    : output_axes_order,
                  'default_image_axes_order' : ['horizontal_x','horizontal_y','vertical'],
                  'default_acquisition_axes_order' : ['angle','vertical','horizontal'] 
                  }
        
        super(CCPiForwardProjector, self).__init__(**kwargs)
        
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
            # sort in the order that this projector needs it
            return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

    def process(self):
        
        volume = self.get_input()
        volume_axes = volume.get_data_axes_order(new_order=self.default_image_axes_order)
        if not volume_axes == [0,1,2]:
            volume.array = numpy.transpose(volume.array, volume_axes)
        pixel_per_voxel = 1 # should be estimated from image_geometry and 
                            # acquisition_geometry
        if self.acquisition_geometry.geom_type == 'parallel':
            #int msize = ndarray_volume.shape(0) > ndarray_volume.shape(1) ? ndarray_volume.shape(0) : ndarray_volume.shape(1);
        	  #int detector_width = msize;
            # detector_width is the max between the shape[0] and shape[1]
            
            
            #double rotation_center = (double)detector_width/2.;
        	  #int detector_height = ndarray_volume.shape(2);
        	  
            #int number_of_projections = ndarray_angles.shape(0);
        	
            ##numpy_3d pixels(reinterpret_cast<float*>(ndarray_volume.get_data()),
		     #boost::extents[number_of_projections][detector_height][detector_width]);

            pixels = pbalg.pb_forward_project(volume.as_array(), 
                                                  self.acquisition_geometry.angles, 
                                                  pixel_per_voxel)
            out = AcquisitionData(geometry=self.acquisition_geometry, 
                                  label_dimensions=self.default_acquisition_axes_order)
            out.fill(pixels)
            out_axes = out.get_data_axes_order(new_order=self.output_axes_order)
            if not out_axes == [0,1,2]:
                out.array = numpy.transpose(out.array, out_axes)
            return out
        else:
            raise ValueError('Cannot process cone beam')

class CCPiBackwardProjector(DataProcessor):
    '''Backward projector
    
    This processor reads in a AcquisitionData and performs a backward projection, 
    i.e. project to reconstruction space.
    Notice that it assumes that the center of rotation is in the middle
    of the horizontal axis: in case when that's not the case it can be chained 
    with the AcquisitionDataPadder.
    
    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSetn
    '''
    
    def __init__(self, 
                 image_geometry = None, 
                 acquisition_geometry = None,
                 output_axes_order=None):
        if output_axes_order is None:
            # default ccpi projector image storing order
            output_axes_order = ['horizontal_x','horizontal_y','vertical']
        kwargs = {
                  'image_geometry'       : image_geometry, 
                  'acquisition_geometry' : acquisition_geometry,
                  'output_axes_order'    : output_axes_order,
                  'default_image_axes_order' : ['horizontal_x','horizontal_y','vertical'],
                  'default_acquisition_axes_order' : ['angle','vertical','horizontal'] 
                  }
        
        super(CCPiBackwardProjector, self).__init__(**kwargs)
        
    def check_input(self, dataset):
        if dataset.number_of_dimensions == 3 or dataset.number_of_dimensions == 2:
            #number_of_projections][detector_height][detector_width
            
            return True
        else:
            raise ValueError("Expected input dimensions is 2 or 3, got {0}"\
                             .format(dataset.number_of_dimensions))

    def process(self):
        projections = self.get_input()
        projections_axes = projections.get_data_axes_order(new_order=self.default_acquisition_axes_order)
        if not projections_axes == [0,1,2]:
            projections.array = numpy.transpose(projections.array, projections_axes)
        
        pixel_per_voxel = 1 # should be estimated from image_geometry and acquisition_geometry
        image_geometry = ImageGeometry(voxel_num_x = self.acquisition_geometry.pixel_num_h,
                                       voxel_num_y = self.acquisition_geometry.pixel_num_h,
                                       voxel_num_z = self.acquisition_geometry.pixel_num_v)
        # input centered/padded acquisitiondata
        center_of_rotation = projections.get_dimension_size('horizontal') / 2
        #print (center_of_rotation)
        if self.acquisition_geometry.geom_type == 'parallel':
            back = pbalg.pb_backward_project(
                         projections.as_array(), 
                         self.acquisition_geometry.angles, 
                         center_of_rotation, 
                         pixel_per_voxel
                         )
            out = ImageData(geometry=self.image_geometry, 
                            dimension_labels=self.default_image_axes_order)
            
            out_axes = out.get_data_axes_order(new_order=self.output_axes_order)
            if not out_axes == [0,1,2]:
                back = numpy.transpose(back, out_axes)
            out.fill(back)
            
            return out
            
        else:
            raise ValueError('Cannot process cone beam')
            
