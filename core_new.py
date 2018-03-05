#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 8 2018

@author: Jens Kammerer
"""


import sys
sys.path.append('/home/kjens/Python/Packages/opticstools/opticstools/opticstools')
sys.path.append('/home/kjens/Python/Development/NACO/kernel/xara')

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import opticstools as ot
import os
import scipy.ndimage as nd
from xml.dom import minidom

make_plots = False
block_plots = False


class cube():
    """
    Make cleaned data cube from ESO xml file
    """
    
    
    def __init__(self,
                 ddir,
                 rdir,
                 cdir,
                 xml_path,
                 shift_as_prior=True):
        """
        Make cleaned data cube from ESO xml file
        
        Parameters
        ----------
        ddir : str
            path of data directory
        rdir : str
            path of reduction directory
        cdir : str
            path of cube directory
        xml_path : str
            path of ESO xml file (must be sub-directory of ddir)
        shift_as_prior : bool
            use 'ESO SEQ CUMOFFSETX' and 'ESO SEQ CUMOFFSETY' as prior for target coordinates
        """
        
        print('--> Initializing directories')
        
        # Path of data directory
        self.ddir = ddir
        
        # Path of reduction directory
        self.rdir = rdir
        
        # Path of cube directory
        self.cdir = cdir
        
        print('--> Reading dark, flat and data paths from ESO xml file')
        
        # Read dark, flat and data paths from ESO xml file
        dark_paths, flat_paths, data_paths = self._read_from_xml(xml_path)
        print('--------------------')
        
        print('--> Making darks')
        
        # Make darks
        if (len(dark_paths) != 0):
            self.dark = dark(self.ddir,
                             self.rdir,
                             dark_paths)
        print('--------------------')
        
        print('--> Making flats')
        
        # Make flats
        if (len(flat_paths) != 0):
            self.flat = flat(self.ddir,
                             self.rdir,
                             flat_paths)
        print('--------------------')
        
        print('--> Making data cubes')
        
        # Make data cubes
        if (len(data_paths) % 2 != 0):
            raise UserWarning('Background subtraction requires even number of data cubes')
        for i in range(0, int(len(data_paths)/2.)):
            self.data = self._clean_data(data_paths[i],
                                         data_paths[int(len(data_paths)/2.)+i],
                                         shift_as_prior=shift_as_prior)
        print('--------------------')
    
    
    def _read_from_xml(self,
                       xml_path):
        """
        Read dark, flat and data paths from ESO xml file
        
        Parameters
        ----------
        xml_path : str
            path of ESO xml file (must be sub-directory of ddir)
        
        Returns
        -------
        dark_paths : list
            list of dark paths
        flat_paths : list
            list of flat paths
        data_paths : list
            list of data paths
        """
        
        # Open ESO xml file
        xml_file = minidom.parse(self.ddir+xml_path)
        
        # Get all items of type file
        xml_items = xml_file.getElementsByTagName('file')
        
        # Identify data, dark and flat items
        dark_paths = []
        flat_paths = []
        data_paths = []
        for item in xml_items:
            if (item.attributes['category'].value == 'IM_JITTER_OBJ'):
                data_paths += [item.attributes['name'].value]
            elif (item.attributes['category'].value == 'CAL_DARK'):
                dark_paths += [item.attributes['name'].value]
            elif (item.attributes['category'].value == 'CAL_FLAT_TW'):
                flat_paths += [item.attributes['name'].value]
            else:
                print('Could not match '+item.attributes['category'].value)
        
        print(str(len(dark_paths))+' dark files identified')
        print(str(len(flat_paths))+' flat files identified')
        print(str(len(data_paths))+' data files identified')
        
        file = open('log.txt', 'a')
        file.write(str(len(dark_paths))+' dark files identified\n')
        file.write(str(len(flat_paths))+' flat files identified\n')
        file.write(str(len(data_paths))+' data files identified\n')
        file.close()
        
        # Return dark, flat and data paths
        return dark_paths, flat_paths, data_paths
    
    
    def _clean_data(self,
                    data_path_1,
                    data_path_2,
                    shift_as_prior=True):
        """
        Make data cube
        
        Parameters
        ----------
        data_path_1 : str
            path of single data cube (must be sub-directory of ddir)
        data_path_2 : str
            path of single data cube (must be sub-directory of ddir)
        shift_as_prior : bool
            use 'ESO SEQ CUMOFFSETX' and 'ESO SEQ CUMOFFSETY' as prior for target coordinates
        """
        
        """
        Process first data cube
        """
        
        # Get data cube 1
        data_path = data_path_1
        try:
            fits_file = pyfits.open(self.ddir+data_path+'.fits')
        except:
            try:
                data_path = data_path.replace(':', '_')
                fits_file = pyfits.open(self.ddir+data_path+'.fits')
            except:
                raise UserWarning('Could not find data cube '+data_path)
        fits_header = fits_file[0].header
        arrays = fits_file[0].data[:-1, :, :] # Last frame is mean, throw it away
        arrays = arrays[1:, :, :]             # First frame is rubbish, throw it away
        fits_file.close()
        print('Loaded data cube '+data_path+' of shape '+str(arrays.shape))
        
        if (arrays.shape[0] < 3):
            return None
        
        # Reshape data cubes with short exposure times to make target visible
        if (fits_header['EXPTIME'] < 0.1):
            print('Reshaping data cube due to short exposure times')
            dummy = arrays.copy()
            arrays = np.zeros((int(dummy.shape[0]/9), dummy.shape[1], dummy.shape[2]))
            for i in range(0, int(dummy.shape[0]/9)):
                arrays[i, :, :] = np.sum(dummy[i*9:i*9+9, :, :], axis=0)
            fits_header['EXPTIME'] = fits_header['EXPTIME']*9.
            print('Data cube was reshaped to shape '+str(arrays.shape))
            print('Lost '+str(dummy.shape[0] % 9)+' files due to stacking')
        
        # Find appropriate master dark and master flat
        dark_path = self._get_dark(fits_header)
        dark_header = pyfits.getheader(self.rdir+dark_path)
        ratio = float(fits_header['EXPTIME'])/float(dark_header['EXPTIME'])
        dark = pyfits.getdata(self.rdir+dark_path, 0)*ratio
        flat_path = self._get_flat(fits_header)
#        flat_header = pyfits.getheader(self.rdir+flat_path)
#        ratio = float(fits_header['EXPTIME'])/float(flat_header['EXPTIME'])
        flat = pyfits.getdata(self.rdir+flat_path, 0)
        bad_pixels = pyfits.getdata(self.rdir+flat_path, 1)
        
        sub_size = 128
        
        # Make Fourier mask
        print('Making pupil mask')
        pmask = self._make_pmask(fits_header, sub_size)
        print('Making Fourier mask')
        fmask = self._pmask_to_fmask(pmask)
        
        # Get info from header
        rot_start = float(fits_header['ESO ADA ABSROT START'])
        rot_end = float(fits_header['ESO ADA ABSROT END'])
        parang_start = float(fits_header['ESO TEL PARANG START'])
        parang_end = float(fits_header['ESO TEL PARANG END'])
        alt = float(fits_header['ESO TEL ALT'])
        instrument_offset = -0.55
        offset_x = int(fits_header['ESO SEQ CUMOFFSETX'])
        offset_y = int(fits_header['ESO SEQ CUMOFFSETY'])
        mode = fits_header['HIERARCH ESO DET MODE NAME']
        if (mode == 'HighWellDepth'):
            gain = 9.8
            read_noise = 4.4
            print('Detector mode is '+mode+', gain = '+str(gain)+', read noise = '+str(read_noise))
        else:
            raise UserWarning('Detector mode '+mode+' is not known')
        
        # Clean each frame individually
        parangs = np.zeros(arrays.shape[0])
        backgrounds = np.zeros(arrays.shape[0])
        for i in range(arrays.shape[0]):
            if ((i+1)%10 == 0):
                print('Cleaning frame '+str(i+1))
            frame = arrays[i, :, :]
            
            # Get parallactic angle
            parangs[i] = alt+instrument_offset+\
                         (rot_start+(rot_end-rot_start)/float(arrays.shape[0])*float(i))-\
                         (180.-(parang_start+(parang_end-parang_start)/float(arrays.shape[0])*float(i))) # FIX
            
            # Subtract master dark and normalize by master flat
            frame = np.true_divide(frame-dark, flat)
            
            if (i == 0):
                out_name = 'dark_flat.png'
                same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                out_name = 'c_%03d' % len(same_tar)+'_'+out_name
                plt.figure()
                plt.imshow(frame)
                plt.savefig(self.rdir+out_name)
                plt.close()
            
            # DODGY: Replacing bad pixels with median filter of size 5
            for j in range(0, 1):
                frame[bad_pixels == 1] = nd.median_filter(frame, size=5)[bad_pixels == 1]
            
            if (i == 0):
                out_name = 'dark_flat_bad.png'
                same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                out_name = 'c_%03d' % len(same_tar)+'_'+out_name
                plt.figure()
                plt.imshow(frame)
                plt.savefig(self.rdir+out_name)
                plt.close()
            
            # Subtract background
            backgrounds[i] = np.median(frame[bad_pixels == 0])
            frame -= backgrounds[i]
            
            if (i == 0):
                out_name = 'dark_flat_bad_bg.png'
                same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                out_name = 'c_%03d' % len(same_tar)+'_'+out_name
                plt.figure()
                plt.imshow(frame, vmin=-50, vmax=50)
                plt.savefig(self.rdir+out_name)
                plt.close()
            
            arrays[i, :, :] = frame
        
        # Shift frames and calculate sum
        shift = np.array([offset_x, offset_y])
        sum_frame = np.zeros((arrays.shape[1], arrays.shape[2]))
        for i in range(0, arrays.shape[0]):
            sum_frame += np.roll(np.roll(arrays[i, :, :], -shift[0], axis=1), -shift[1], axis=0) # Axes are swapped between ESO and Python
        
        # Find target
        if (shift_as_prior == True):
            x_prior = int(sum_frame.shape[0]/2.)
            y_prior = int(sum_frame.shape[1]/2.)
            
            if (make_plots == True):
                plt.figure()
                plt.imshow(sum_frame)
                plt.show(block=block_plots)
                plt.close()
            
            sum_frame = sum_frame[x_prior-32:x_prior+32, y_prior-32:y_prior+32]
            sum_frame_filtered = nd.filters.median_filter(sum_frame, size=5)
            max_index = np.unravel_index(sum_frame_filtered.argmax(), sum_frame_filtered.shape)
            print('Maximum x, y = '+str(max_index[1]+(y_prior-32))+', '+str(max_index[0]+(x_prior-32)))
            max_x = max_index[1]+(y_prior-32)+shift[0]
            max_y = max_index[0]+(x_prior-32)+shift[1]
        else:
            sum_frame_filtered = nd.filters.median_filter(sum_frame, size=5)
            max_index = np.unravel_index(sum_frame_filtered.argmax(), sum_frame_filtered.shape)
            print('Maximum x, y = '+str(max_index[1])+', '+str(max_index[0]))
            max_x = max_index[1]+shift[0]
            max_y = max_index[0]+shift[1]
        
        arrays_1 = arrays.copy(); del arrays
        backgrounds_1 = backgrounds.copy(); del backgrounds
        bad_pixels_1 = bad_pixels.copy(); del bad_pixels
        fits_header_1 = fits_header.copy(); del fits_header
        fmask_1 = fmask.copy(); del fmask
        max_x_1 = max_x.copy(); del max_x
        max_y_1 = max_y.copy(); del max_y
        parangs_1 = parangs.copy(); del parangs
        
        """
        Process second data cube
        """
        
        # Get data cube 2
        data_path = data_path_2
        try:
            fits_file = pyfits.open(self.ddir+data_path+'.fits')
        except:
            try:
                data_path = data_path.replace(':', '_')
                fits_file = pyfits.open(self.ddir+data_path+'.fits')
            except:
                raise UserWarning('Could not find data cube '+data_path)
        fits_header = fits_file[0].header
        arrays = fits_file[0].data[:-1, :, :] # Last frame is mean, throw it away
        arrays = arrays[1:, :, :] # First frame is rubbish, throw it away
        fits_file.close()
        print('Loaded data cube '+data_path+' of shape '+str(arrays.shape))
        
        if (arrays.shape[0] < 3):
            return None
        
        # Reshape data cubes with short exposure times to make target visible
        if (fits_header['EXPTIME'] < 0.1):
            print('Reshaping data cube due to short exposure times')
            dummy = arrays.copy()
            arrays = np.zeros((int(dummy.shape[0]/9), dummy.shape[1], dummy.shape[2]))
            for i in range(0, int(dummy.shape[0]/9)):
                arrays[i, :, :] = np.sum(dummy[i*9:i*9+9, :, :], axis=0)
            fits_header['EXPTIME'] = fits_header['EXPTIME']*9.
            print('Data cube was reshaped to shape '+str(arrays.shape))
            print('Lost '+str(dummy.shape[0] % 9)+' files due to stacking')
        
        # Find appropriate master dark and master flat
        dark_path = self._get_dark(fits_header)
        dark_header = pyfits.getheader(self.rdir+dark_path)
        ratio = float(fits_header['EXPTIME'])/float(dark_header['EXPTIME'])
        dark = pyfits.getdata(self.rdir+dark_path, 0)*ratio
        flat_path = self._get_flat(fits_header)
#        flat_header = pyfits.getheader(self.rdir+flat_path)
#        ratio = float(fits_header['EXPTIME'])/float(flat_header['EXPTIME'])
        flat = pyfits.getdata(self.rdir+flat_path, 0)
        bad_pixels = pyfits.getdata(self.rdir+flat_path, 1)
        
        sub_size = 128
        
        # Make Fourier mask
        print('Making pupil mask')
        pmask = self._make_pmask(fits_header, sub_size)
        print('Making Fourier mask')
        fmask = self._pmask_to_fmask(pmask)
        
        # Get info from header
        rot_start = float(fits_header['ESO ADA ABSROT START'])
        rot_end = float(fits_header['ESO ADA ABSROT END'])
        parang_start = float(fits_header['ESO TEL PARANG START'])
        parang_end = float(fits_header['ESO TEL PARANG END'])
        alt = float(fits_header['ESO TEL ALT'])
        instrument_offset = -0.55
        offset_x = int(fits_header['ESO SEQ CUMOFFSETX'])
        offset_y = int(fits_header['ESO SEQ CUMOFFSETY'])
        mode = fits_header['HIERARCH ESO DET MODE NAME']
        if (mode == 'HighWellDepth'):
            gain = 9.8
            read_noise = 4.4
            print('Detector mode is '+mode+', gain = '+str(gain)+', read noise = '+str(read_noise))
        else:
            raise UserWarning('Detector mode '+mode+' is not known')
        
        # Clean each frame individually
        parangs = np.zeros(arrays.shape[0])
        backgrounds = np.zeros(arrays.shape[0])
        for i in range(arrays.shape[0]):
            if ((i+1)%10 == 0):
                print('Cleaning frame '+str(i+1))
            frame = arrays[i, :, :]
            
            # Get parallactic angle
            parangs[i] = alt+instrument_offset+\
                         (rot_start+(rot_end-rot_start)/float(arrays.shape[0])*float(i))-\
                         (180.-(parang_start+(parang_end-parang_start)/float(arrays.shape[0])*float(i))) # FIX
            
            # Subtract master dark and normalize by master flat
            frame = np.true_divide(frame-dark, flat)
            
            if (i == 0):
                out_name = 'dark_flat.png'
                same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                out_name = 'c_%03d' % len(same_tar)+'_'+out_name
                plt.figure()
                plt.imshow(frame)
                plt.savefig(self.rdir+out_name)
                plt.close()
            
            # DODGY: Replacing bad pixels with median filter of size 5
            for j in range(0, 1):
                frame[bad_pixels == 1] = nd.median_filter(frame, size=5)[bad_pixels == 1]
            
            if (i == 0):
                out_name = 'dark_flat_bad.png'
                same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                out_name = 'c_%03d' % len(same_tar)+'_'+out_name
                plt.figure()
                plt.imshow(frame)
                plt.savefig(self.rdir+out_name)
                plt.close()
            
            # Subtract background
            backgrounds[i] = np.median(frame[bad_pixels == 0])
            frame -= backgrounds[i]
            
            if (i == 0):
                out_name = 'dark_flat_bad_bg.png'
                same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                out_name = 'c_%03d' % len(same_tar)+'_'+out_name
                plt.figure()
                plt.imshow(frame, vmin=-50, vmax=50)
                plt.savefig(self.rdir+out_name)
                plt.close()
            
            arrays[i, :, :] = frame
        
        # Shift frames and calculate sum
        shift = np.array([offset_x, offset_y])
        sum_frame = np.zeros((arrays.shape[1], arrays.shape[2]))
        for i in range(0, arrays.shape[0]):
            sum_frame += np.roll(np.roll(arrays[i, :, :], -shift[0], axis=1), -shift[1], axis=0) # Axes are swapped between ESO and Python
        
        # Find target
        if (shift_as_prior == True):
            x_prior = int(sum_frame.shape[0]/2.)
            y_prior = int(sum_frame.shape[1]/2.)
            
            if (make_plots == True):
                plt.figure()
                plt.imshow(sum_frame)
                plt.show(block=block_plots)
                plt.close()
            
            sum_frame = sum_frame[x_prior-32:x_prior+32, y_prior-32:y_prior+32]
            sum_frame_filtered = nd.filters.median_filter(sum_frame, size=5)
            max_index = np.unravel_index(sum_frame_filtered.argmax(), sum_frame_filtered.shape)
            print('Maximum x, y = '+str(max_index[1]+(y_prior-32))+', '+str(max_index[0]+(x_prior-32)))
            max_x = max_index[1]+(y_prior-32)+shift[0]
            max_y = max_index[0]+(x_prior-32)+shift[1]
        else:
            sum_frame_filtered = nd.filters.median_filter(sum_frame, size=5)
            max_index = np.unravel_index(sum_frame_filtered.argmax(), sum_frame_filtered.shape)
            print('Maximum x, y = '+str(max_index[1])+', '+str(max_index[0]))
            max_x = max_index[1]+shift[0]
            max_y = max_index[0]+shift[1]
        
        arrays_2 = arrays.copy(); del arrays
        backgrounds_2 = backgrounds.copy(); del backgrounds
        bad_pixels_2 = bad_pixels.copy(); del bad_pixels
        fits_header_2 = fits_header.copy(); del fits_header
        fmask_2 = fmask.copy(); del fmask
        max_x_2 = max_x.copy(); del max_x
        max_y_2 = max_y.copy(); del max_y
        parangs_2 = parangs.copy(); del parangs
        
        """
        Advanced background subtraction
        """
        
        med_1 = np.median(arrays_1, axis=0)
        med_2 = np.median(arrays_2, axis=0)
        
        for i in range(0, arrays_1.shape[0]):
            arrays_1[i, :, :] -= med_2
        for i in range(0, arrays_2.shape[0]):
            arrays_2[i, :, :] -= med_1
        
        out_name = 'dark_flat_bad_bg_adv.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(arrays_1[0], vmin=-50, vmax=50)
        plt.savefig(self.rdir+out_name)
        plt.close()
        
        out_name = 'dark_flat_bad_bg_adv.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(arrays_2[0], vmin=-50, vmax=50)
        plt.savefig(self.rdir+out_name)
        plt.close()
        
        """
        Check that both data cubes come from same OB
        """
        
        if (not np.array_equal(bad_pixels_1, bad_pixels_2)):
            raise UserWarning('Cubes do not come from same OB')
        else:
            bad_pixels = bad_pixels_1.copy(); del bad_pixels_1; del bad_pixels_2
        if (not np.array_equal(fmask_1, fmask_2)):
            raise UserWarning('Cubes do not come from same OB')
        else:
            fmask = fmask_1.copy(); del fmask_1; del fmask_2
        
        if (make_plots == True):
            plt.figure()
            plt.imshow(bad_pixels)
            plt.show(block=block_plots)
            plt.close()
        
        # Remove stripes from bad pixel map
        bad_pixels, stripes_map = self._remove_stripes(bad_pixels)
        
        if (make_plots == True):
            plt.figure()
            plt.imshow(bad_pixels)
            plt.show(block=block_plots)
            plt.close()
        
        """
        Process first data cube
        """
        
        # Calculate median frame (noise reduced)
        frame = np.roll(np.roll(np.median(arrays_1, axis=0),\
                int(sub_size/2.)-max_y_1, axis=0),\
                int(sub_size/2.)-max_x_1, axis=1)[0:sub_size, 0:sub_size]
        
        out_name = 'final_power_bin.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(frame < -100)
        plt.savefig(self.rdir+out_name)
        plt.show(block=block_plots)
        
        # Check if PSF from dithering subtraction overlaps with sub-array
        if (np.sum(frame < -100) > 10): # USER
            sub_size = 80
            frame = np.roll(np.roll(arrays_1[0],\
                    int(sub_size/2.)-max_y_1, axis=0),\
                    int(sub_size/2.)-max_x_1, axis=1)[0:sub_size, 0:sub_size]
            
            # Re-make Fourier mask
            print('Re-making pupil mask')
            pmask = self._make_pmask(fits_header_1, sub_size)
            print('Re-making Fourier mask')
            fmask = self._pmask_to_fmask(pmask)
        
        out_name = 'final_power.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(frame, vmin=-100, vmax=100)
        plt.savefig(self.rdir+out_name)
        plt.show(block=block_plots)
        
        file = open('log.txt', 'a')
        file.write('Data cube shape = '+str(arrays_1.shape)+', sub-array size = '+str(sub_size)+'\n')
        file.close()
        
        # Cut out sub-arrays
        print('Cutting out sub-arrays')
        sub_arrays = np.zeros((arrays_1.shape[0], sub_size, sub_size))
        sub_arrays_bad_pixels = np.zeros((arrays_1.shape[0], sub_size, sub_size))
        sub_stripes_map = np.roll(np.roll(stripes_map,\
                                          int(sub_size/2.)-max_y_1, axis=0),\
                                          int(sub_size/2.)-max_x_1, axis=1)[0:sub_size, 0:sub_size]
        
        out_name = 'sub_stripes_map.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(sub_stripes_map)
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        # Remove bad pixels
        maxima = np.zeros(arrays_1.shape[0])
        for i in range(0, arrays_1.shape[0]):
            print('Removing bad pixels from frame '+str(i+1))
            frame = arrays_1[i, :, :]
            sub_arrays[i, :, :] = np.roll(np.roll(frame,\
                                  int(sub_size/2.)-max_y_1, axis=0),\
                                  int(sub_size/2.)-max_x_1, axis=1)[0:sub_size, 0:sub_size]
            sub_bad_pixels = np.roll(np.roll(bad_pixels,\
                             int(sub_size/2.)-max_y_1, axis=0),\
                             int(sub_size/2.)-max_x_1, axis=1)[0:sub_size, 0:sub_size]
            
            # Find maximum of frame
            frame_filtered = nd.filters.median_filter(sub_arrays[i, :, :], size=5)
            max_index = np.unravel_index(frame_filtered.argmax(), frame_filtered.shape)
            maxima[i] = sub_arrays[i, max_index[0], max_index[1]]
            
            # Find bad pixels frame
            new_bad_pixels = sub_bad_pixels.copy()
            dummy1 = sub_arrays[i, :, :].copy()
            dummy1[np.where(sub_bad_pixels)] = 0.
            
            # Find saturated pixels
            thresh = 17000.-backgrounds_1[i]
            sub_bad_pixels[self._saturated_pixels(dummy1, thresh=thresh)] = 1
            
            if (i == 0 and make_plots == True):
                plt.figure()
                plt.imshow(dummy1 > thresh)
                plt.show(block=block_plots)
                plt.close()
            
            for loops in range(1, 15):
                
                # Correct known bad pixels
                dummy2 = self._fix_bad_pixels(dummy1, sub_bad_pixels, fmask)
                
                # Find extra bad pixels
                extra_frame_ft = np.fft.rfft2(dummy2)*fmask
                extra_frame = np.real(np.fft.irfft2(extra_frame_ft))
                frame_median = nd.filters.median_filter(dummy2, size=5)
                
                # Using gain and read noise from NACO user manual
                total_noise = np.sqrt(np.maximum((backgrounds_1[i]+frame_median)/gain+read_noise**2,\
                                                 read_noise**2))
                extra_frame /= total_noise
                
                # Subtract median filtered frame (ringing effect)
                unsharp_masked = extra_frame-nd.filters.median_filter(extra_frame, size=3)
                
                # Find extra bad pixels based on variable threshold
                extra_threshold = 7 # USER
                current_threshold = np.max([0.25*np.max(np.abs(unsharp_masked[new_bad_pixels < 0.5])),\
                                            extra_threshold*np.median(np.abs(extra_frame))]) # Pixels above 1/4 of the maximum or above 7 times the median are bad
                extra_bad_pixels = np.abs(unsharp_masked) > current_threshold
                #extra_bad_pixels[np.where(sub_stripes_map > 0.5)] = 0
                n_extra_bad_pixels = np.sum(extra_bad_pixels)
                print(str(n_extra_bad_pixels)+' extra bad pixels identified (attempt '+str(loops)+'), threshold = '+str(current_threshold))
                
                # Add extra bad pixels to bad pixel map
                sub_bad_pixels += extra_bad_pixels
                sub_bad_pixels = sub_bad_pixels > 0.5
                new_bad_pixels = extra_bad_pixels > 0.5
                
                # Break if no extra bad pixels were found
                if (n_extra_bad_pixels == 0):
                    break
            print(str(np.sum(sub_bad_pixels))+' total bad pixels identified')
            
            # Correct all bad pixels
            sub_arrays[i, :, :] = self._fix_bad_pixels(sub_arrays[i, :, :], sub_bad_pixels, fmask)
            sub_arrays_bad_pixels[i, :, :] = sub_bad_pixels
        
        # Remove frames with low peak count
        median_cut = 0.7
        high_count = np.where(maxima > median_cut*np.median(maxima))
        high_count = high_count[0]
        print(str(sub_arrays.shape[0]-len(high_count))+' frames rejected due to low peak count')
        sub_arrays = sub_arrays[high_count, :, :]
        
        out_name = 'final_median.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(np.median(sub_arrays, axis=0))
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        out_name = 'final_bad_pixels.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(np.sum(sub_arrays_bad_pixels, axis=0))
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        # Save data cube
        fits_header_1['NAXIS2'] = sub_arrays.shape[1]
        fits_header_1['NAXIS1'] = sub_arrays.shape[2]
        fits_header_1['NAXIS3'] = sub_arrays.shape[0]
        out_name = str(fits_header_1['HIERARCH ESO OBS TARG NAME'])+'_'+\
                   str(fits_header_1['NAXIS2'])+'_'+\
                   str(fits_header_1['NAXIS1'])+'_'+\
                   str(fits_header_1['EXPTIME'])[0:4]+'_'+\
                   str(fits_header_1['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                   str(fits_header_1['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                   str(fits_header_1['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
        same_tar = [f for f in os.listdir(self.cdir) if (out_name in f)]
        out_name = '%03d' % len(same_tar)+'_'+out_name
        out_file = pyfits.HDUList()
        out_file.append(pyfits.ImageHDU(sub_arrays, fits_header_1))
#        col1 = pyfits.Column(name='xpeak', format='E', array=max_x)
#        col2 = pyfits.Column(name='ypeak', format='E', array=max_y)
        col3 = pyfits.Column(name='pa', format='E', array=parangs_1)
        col4 = pyfits.Column(name='max', format='E', array=maxima)
        col5 = pyfits.Column(name='background', format='E', array=backgrounds_1)
        cols = pyfits.ColDefs([col3, col4, col5])
        out_file.append(pyfits.BinTableHDU.from_columns(cols))
        out_file.append(pyfits.ImageHDU(np.uint8(sub_arrays_bad_pixels)))
        out_file.writeto(self.cdir+out_name+'.fits',\
                         output_verify='ignore',\
                         overwrite=True)
        
        file = open('log.txt', 'a')
        file.write('Saved as '+out_name+'.fits\n')
        file.close()
        
        """
        Process second data cube
        """
        
        file = open('log.txt', 'a')
        file.write('Data cube shape = '+str(arrays_2.shape)+', sub-array size = '+str(sub_size)+'\n')
        file.close()
        
        # Cut out sub-arrays
        print('Cutting out sub-arrays')
        sub_arrays = np.zeros((arrays_2.shape[0], sub_size, sub_size))
        sub_arrays_bad_pixels = np.zeros((arrays_2.shape[0], sub_size, sub_size))
        sub_stripes_map = np.roll(np.roll(stripes_map,\
                                          int(sub_size/2.)-max_y_2, axis=0),\
                                          int(sub_size/2.)-max_x_2, axis=1)[0:sub_size, 0:sub_size]
        
        out_name = 'sub_stripes_map.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(sub_stripes_map)
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        # Remove bad pixels
        maxima = np.zeros(arrays_2.shape[0])
        for i in range(0, arrays_2.shape[0]):
            print('Removing bad pixels from frame '+str(i+1))
            frame = arrays_2[i, :, :]
            sub_arrays[i, :, :] = np.roll(np.roll(frame,\
                                  int(sub_size/2.)-max_y_2, axis=0),\
                                  int(sub_size/2.)-max_x_2, axis=1)[0:sub_size, 0:sub_size]
            sub_bad_pixels = np.roll(np.roll(bad_pixels,\
                             int(sub_size/2.)-max_y_2, axis=0),\
                             int(sub_size/2.)-max_x_2, axis=1)[0:sub_size, 0:sub_size]
        
            # Find maximum of frame
            frame_filtered = nd.filters.median_filter(sub_arrays[i, :, :], size=5)
            max_index = np.unravel_index(frame_filtered.argmax(), frame_filtered.shape)
            maxima[i] = sub_arrays[i, max_index[0], max_index[1]]
            
            # Find bad pixels frame
            new_bad_pixels = sub_bad_pixels.copy()
            dummy1 = sub_arrays[i, :, :].copy()
            dummy1[np.where(sub_bad_pixels)] = 0.
            
            # Find saturated pixels
            thresh = 17000.-backgrounds_2[i]
            sub_bad_pixels[self._saturated_pixels(dummy1, thresh=thresh)] = 1
            
            if (i == 0 and make_plots == True):
                plt.figure()
                plt.imshow(dummy1 > thresh)
                plt.show(block=block_plots)
                plt.close()
            
            for loops in range(1, 15):
                
                # Correct known bad pixels
                dummy2 = self._fix_bad_pixels(dummy1, sub_bad_pixels, fmask)
                
                # Find extra bad pixels
                extra_frame_ft = np.fft.rfft2(dummy2)*fmask
                extra_frame = np.real(np.fft.irfft2(extra_frame_ft))
                frame_median = nd.filters.median_filter(dummy2, size=5)
                
                # Using gain and read noise from NACO user manual
                total_noise = np.sqrt(np.maximum((backgrounds_2[i]+frame_median)/gain+read_noise**2,\
                                                 read_noise**2))
                extra_frame /= total_noise
                
                # Subtract median filtered frame (ringing effect)
                unsharp_masked = extra_frame-nd.filters.median_filter(extra_frame, size=3)
                
                # Find extra bad pixels based on variable threshold
                extra_threshold = 7 # USER
                current_threshold = np.max([0.25*np.max(np.abs(unsharp_masked[new_bad_pixels < 0.5])),\
                                            extra_threshold*np.median(np.abs(extra_frame))]) # Pixels above 1/4 of the maximum or above 7 times the median are bad
                extra_bad_pixels = np.abs(unsharp_masked) > current_threshold
                extra_bad_pixels[np.where(sub_stripes_map > 0.5)] = 0
                n_extra_bad_pixels = np.sum(extra_bad_pixels)
                print(str(n_extra_bad_pixels)+' extra bad pixels identified (attempt '+str(loops)+'), threshold = '+str(current_threshold))
                
                # Add extra bad pixels to bad pixel map
                sub_bad_pixels += extra_bad_pixels
                sub_bad_pixels = sub_bad_pixels > 0.5
                new_bad_pixels = extra_bad_pixels > 0.5
                
                # Break if no extra bad pixels were found
                if (n_extra_bad_pixels == 0):
                    break
            print(str(np.sum(sub_bad_pixels))+' total bad pixels identified')
            
            # Correct all bad pixels
            sub_arrays[i, :, :] = self._fix_bad_pixels(sub_arrays[i, :, :], sub_bad_pixels, fmask)
            sub_arrays_bad_pixels[i, :, :] = sub_bad_pixels
        
        # Remove frames with low peak count
        median_cut = 0.7
        high_count = np.where(maxima > median_cut*np.median(maxima))
        high_count = high_count[0]
        print(str(sub_arrays.shape[0]-len(high_count))+' frames rejected due to low peak count')
        sub_arrays = sub_arrays[high_count, :, :]
        
        out_name = 'final_median.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(np.median(sub_arrays, axis=0))
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        out_name = 'final_bad_pixels.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'c_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(np.sum(sub_arrays_bad_pixels, axis=0))
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        # Save data cube
        fits_header_2['NAXIS2'] = sub_arrays.shape[1]
        fits_header_2['NAXIS1'] = sub_arrays.shape[2]
        fits_header_2['NAXIS3'] = sub_arrays.shape[0]
        out_name = str(fits_header_2['HIERARCH ESO OBS TARG NAME'])+'_'+\
                   str(fits_header_2['NAXIS2'])+'_'+\
                   str(fits_header_2['NAXIS1'])+'_'+\
                   str(fits_header_2['EXPTIME'])[0:4]+'_'+\
                   str(fits_header_2['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                   str(fits_header_2['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                   str(fits_header_2['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
        same_tar = [f for f in os.listdir(self.cdir) if (out_name in f)]
        out_name = '%03d' % len(same_tar)+'_'+out_name
        out_file = pyfits.HDUList()
        out_file.append(pyfits.ImageHDU(sub_arrays, fits_header_2))
#        col1 = pyfits.Column(name='xpeak', format='E', array=max_x)
#        col2 = pyfits.Column(name='ypeak', format='E', array=max_y)
        col3 = pyfits.Column(name='pa', format='E', array=parangs_2)
        col4 = pyfits.Column(name='max', format='E', array=maxima)
        col5 = pyfits.Column(name='background', format='E', array=backgrounds_2)
        cols = pyfits.ColDefs([col3, col4, col5])
        out_file.append(pyfits.BinTableHDU.from_columns(cols))
        out_file.append(pyfits.ImageHDU(np.uint8(sub_arrays_bad_pixels)))
        out_file.writeto(self.cdir+out_name+'.fits',\
                         output_verify='ignore',\
                         overwrite=True)
        
        file = open('log.txt', 'a')
        file.write('Saved as '+out_name+'.fits\n')
        file.close()
        
        # Return nothing
        return None
    
    
    def _get_dark(self,
                  fits_header):
        """
        Find appropriate master dark
        
        Parameters
        ----------
        fits_header : header
            header of data cube for which appropriate master dark should be found
        
        Returns
        -------
        dark_path : str
            path of appropriate master dark
        """
        
        # Get all master darks
        master_darks = [f for f in os.listdir(self.rdir) if f.startswith('master_dark')]
        
        # Find appropriate master dark
        flag = False
        for i in range(0, len(master_darks)):
            if (str(fits_header['NAXIS2']) in master_darks[i] and\
                str(fits_header['NAXIS1']) in master_darks[i] and\
                str(fits_header['EXPTIME'])[0:4] in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS CWLEN'])[0:4] in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS OPTI6 ID']) in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                dark_path = master_darks[i]
                flag = True
                print('Found appropriate master dark')
                break
        if (flag == False):
            print('Searching among different wavelengths and filters')
            for i in range(0, len(master_darks)):
                if (str(fits_header['NAXIS2']) in master_darks[i] and\
                    str(fits_header['NAXIS1']) in master_darks[i] and\
                    str(fits_header['EXPTIME'])[0:4] in master_darks[i] and\
                    str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                    dark_path = master_darks[i]
                    flag = True
                    print('Found appropriate master dark')
                    break
        if (flag == False):
            print('Searching among different exposure times')
            for i in range(0, len(master_darks)):
                if (str(fits_header['NAXIS2']) in master_darks[i] and\
                    str(fits_header['NAXIS1']) in master_darks[i] and\
                    str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                    dark_path = master_darks[i]
                    flag = True
                    print('Found appropriate master dark')
                    break
        if (flag == False):
            raise UserWarning('Could not find appropriate master dark')
        
        # Return appropriate master dark
        return dark_path
    
    
    def _get_flat(self,
                  fits_header):
        """
        Find appropriate master flat
        
        Parameters
        ----------
        fits_header : header
            header of data cube for which appropriate master flat should be found
        
        Returns
        -------
        flat_path : str
            path of appropriate master flat
        """
        
        # Get all master darks
        master_flats = [f for f in os.listdir(self.rdir) if f.startswith('master_flat')]
        
        # Find appropriate master flat
        flag = False
        for i in range(0, len(master_flats)):
            if (str(fits_header['NAXIS2']) in master_flats[i] and\
                str(fits_header['NAXIS1']) in master_flats[i] and\
                str(fits_header['EXPTIME'])[0:4] in master_flats[i] and\
                str(fits_header['HIERARCH ESO INS CWLEN'])[0:4] in master_flats[i] and\
                str(fits_header['HIERARCH ESO INS OPTI6 ID']) in master_flats[i] and\
                str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_flats[i]):
                flat_path = master_flats[i]
                flag = True
                print('Found appropriate master flat')
                break
        if (flag == False):
            print('Searching among different shapes and trying to crop')
            for i in range(0, len(master_flats)):
                if (str(fits_header['EXPTIME'])[0:4] in master_flats[i] and\
                    str(fits_header['HIERARCH ESO INS CWLEN'])[0:4] in master_flats[i] and\
                    str(fits_header['HIERARCH ESO INS OPTI6 ID']) in master_flats[i] and\
                    str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_flats[i]):
                    
                    # Crop master flat
                    dummy_file = pyfits.open(self.rdir+master_flats[i])
                    dummy_header = dummy_file[0].header
                    if (fits_header['NAXIS1'] % 2 != 0 or\
                        fits_header['NAXIS2'] % 2 != 0 or\
                        dummy_header['NAXIS1'] % 2 != 0 or\
                        dummy_header['NAXIS2'] % 2 != 0):
                        raise UserWarning('Cropping requires arrays of even shape')
                    med_flat = dummy_file[0].data
                    bad_pixels = dummy_file[1].data
                    x_min = int(med_flat.shape[0]/2.)-int(float(fits_header['NAXIS2'])/2.)
                    y_min = int(med_flat.shape[1]/2.)-int(float(fits_header['NAXIS1'])/2.)
                    med_flat = med_flat[x_min:x_min+int(fits_header['NAXIS2']), y_min:y_min+int(fits_header['NAXIS1'])]
                    bad_pixels = bad_pixels[x_min:x_min+int(fits_header['NAXIS2']), y_min:y_min+int(fits_header['NAXIS1'])]
                    
                    if (make_plots == True):
                        plt.figure()
                        plt.imshow(med_flat)
                        plt.show(block=block_plots)
                        plt.close()
                    
                    if (make_plots == True):
                        plt.figure()
                        plt.imshow(bad_pixels)
                        plt.show(block=block_plots)
                        plt.close()
                    
                    dummy_header['NAXIS1'] = fits_header['NAXIS1']
                    dummy_header['NAXIS2'] = fits_header['NAXIS2']
                    out_name = 'master_flat_'+\
                               str(fits_header['NAXIS2'])+'_'+\
                               str(fits_header['NAXIS1'])+'_'+\
                               str(dummy_header['EXPTIME'])[0:4]+'_'+\
                               str(dummy_header['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                               str(dummy_header['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                               str(dummy_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
                    out_file = pyfits.HDUList()
                    out_file.append(pyfits.ImageHDU(med_flat, dummy_header))
                    out_file.append(pyfits.ImageHDU(bad_pixels))
                    out_file.writeto(self.rdir+out_name+'.fits',\
                                     output_verify='ignore',\
                                     overwrite=True)
                    dummy_file.close()
                    
                    flat_path = out_name+'.fits'
                    
                    flag = True
                    print('Created appropriate master flat')
                    break
        if (flag == False):
            print('Searching among different exposure times')
            for i in range(0, len(master_flats)):
                if (str(fits_header['NAXIS2']) in master_flats[i] and\
                    str(fits_header['NAXIS1']) in master_flats[i] and\
                    str(fits_header['HIERARCH ESO INS CWLEN'])[0:4] in master_flats[i] and\
                    str(fits_header['HIERARCH ESO INS OPTI6 ID']) in master_flats[i] and\
                    str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_flats[i]):
                    flat_path = master_flats[i]
                    flag = True
                    print('Found appropriate master flat')
                    break
        if (flag == False):
            raise UserWarning('Could not find appropriate master flat')
        
        # Return appropriate master flat
        return flat_path
    
    
    def _make_pmask(self,
                    fits_header,
                    sub_size):
        """
        Make pupil mask
        
        Parameters
        ----------
        fits_header : header
            header of data cube for which pupil mask should be made
        sub_size : int
            size of sub-arrays
        
        Returns
        -------
        pmask : array
            pupil mask (boolean)
        """
        
        # Set telescope dimensions
        mirror_size = 8.2 # meters
        
        # Get info from header
        lam = float(fits_header['HIERARCH ESO INS CWLEN'])
        pixscale = float(fits_header['HIERARCH ESO INS PIXSCALE'])
        
        # Set scale ratio for Fourier transformation
        ratio = lam*1E-6/(pixscale/60./60.*np.pi/180.*sub_size)
        
        # Make pupil mask
        pmask = ot.circle(sub_size, mirror_size/ratio)
        
        if (make_plots == True):
            plt.figure()
            plt.imshow(pmask)
            plt.show(block=block_plots)
            plt.close()
        
        # Return pupil mask
        return pmask
    
    
    def _pmask_to_fmask(self,
                        pmask):
        """
        Make Fourier mask
        
        Parameters
        ----------
        pmask : array
            pupil mask (boolean)
        
        Returns
        -------
        fmask : array
            Fourier mask (boolean)
        """
        
        # Find pupil plane positions of pupil mask
        pupil_plane_positions = np.where(pmask > 0.5)
        
        # Spatially ordered matrix containing the baselines contributing to each Fourier plane position
        dummy = np.zeros((2*pmask.shape[0]-1, 2*pmask.shape[1]-1, len(pupil_plane_positions[0])-1))
        
        # Go through all pupil plane baselines
        for i in range(0, len(pupil_plane_positions[0])-1):
            for j in range(i+1, len(pupil_plane_positions[0])):
                
                # Calculate Fourier plane vector corresponding to pupil plane baseline
                dx = pupil_plane_positions[0][j]-pupil_plane_positions[0][i]
                dy = pupil_plane_positions[1][j]-pupil_plane_positions[1][i]
                
                # Project Fourier plane vector onto both half planes
                dx_inv = dx*(-1)
                dy_inv = dy*(-1)
                
                # Basis for z-axis are baselines 01, 02, 03, ...
                if (i == 0):
                    dummy[dx+pmask.shape[0]-1, dy+pmask.shape[1]-1, j-1] += 1
                    dummy[dx_inv+pmask.shape[0]-1, dy_inv+pmask.shape[1]-1, j-1] += 1
                else:
                    dummy[dx+pmask.shape[0]-1, dy+pmask.shape[1]-1, i-1] -= 1
                    dummy[dx_inv+pmask.shape[0]-1, dy_inv+pmask.shape[1]-1, i-1] -= 1
                    dummy[dx+pmask.shape[0]-1, dy+pmask.shape[1]-1, j-1] += 1
                    dummy[dx_inv+pmask.shape[0]-1, dy_inv+pmask.shape[1]-1, j-1] += 1
        
        # Redundancy-coded Fourier mask
        fmask = np.zeros((pmask.shape[0], pmask.shape[1]), dtype='int')
        
        # Find discrete Fourier plane positions
        for i in range(0, dummy.shape[0]):
            for j in range(0, dummy.shape[1]):
                
                # Find redundancy of Fourier plane positions
                redundancy = int((len(np.where(dummy[i, j, :] != 0)[0])+1)/2)
                
                # Check whether redundancy is non-zero
                if (redundancy != 0):
                    
                    # Check whether Fourier plane positions lies inside sub-array size
                    if (i-int(pmask.shape[0]/2) < 0 or\
                        i-int(pmask.shape[0]/2) >= pmask.shape[0] or\
                        j-int(pmask.shape[1]/2) < 0 or\
                        j-int(pmask.shape[1]/2) >= pmask.shape[1]):
                        raise ValueError('Only central quadrant of pmask can be non-zero!')
                    
                    # Fill Fourier mask
                    fmask[i-int(pmask.shape[0]/2), j-int(pmask.shape[1]/2)] = redundancy
        
        # Make Fourier mask rfft compatible
        fmask = np.roll(np.roll(fmask, -int(pmask.shape[0]/2.), axis=0),\
                                       -int(pmask.shape[1]/2.), axis=1)
        fmask = (fmask < 0.5)[:, 0:int(pmask.shape[1]/2.)+1]
        
        if (make_plots == True):
            plt.figure()
            plt.imshow((fmask > 0.5))
            plt.show(block=block_plots)
            plt.close()
        
        # Return Fourier mask
        return fmask
    
    
    def _saturated_pixels(self,
                          frame,
                          thresh):
        """
        Get saturated pixels
        
        Parameters
        ----------
        frame : array
            frame of which saturated pixels should be found
        thresh : float
            threshold above which pixels are regarded as saturated
        
        Returns
        -------
        saturated_pixels : array
            map of saturated pixels (boolean)
        """
        
        # Return map of saturated pixels
        return np.where(frame > thresh)
    
    
    def _remove_stripes(self,
                        bad_pixels):
        """
        Remove stripes from bad pixel map
        
        Parameters
        ----------
        bad_pixels : array
            map of bad pixels (boolean)
        
        Returns
        -------
        bad_pixels : array
            map of bad pixels with stripes removed (boolean)
        stripes_map : array
            map of stripes (boolean)
        """
        
        # Minimal required length of stripe is 1/4 of array size
        threshold = bad_pixels.shape[0]/4.
        
        # Identify stripes
        stripes_map = np.zeros((bad_pixels.shape[0], bad_pixels.shape[1]))
        for i in range(0, bad_pixels.shape[1]):
            if (bad_pixels[0, i] > 0.5):
                for j in range(1, bad_pixels.shape[0]):
                    if (bad_pixels[j, i] < 0.5):
                        break
                if (j >= int(threshold)):
                    bad_pixels[0:j, i] = 0
                    stripes_map[0:j, i] = 1
        
        # Return map of bad pixels with stripes removed and map of stripes
        return bad_pixels, stripes_map
    
    
    def _fix_bad_pixels(self,
                        frame,
                        bad_pixels,
                        fmask):
        """
        Correct bad pixels
        
        Parameters
        ----------
        frame : array
            frame of which bad pixels should be corrected
        bad_pixels : array
            map of bad pixels (boolean)
        fmask : array
            Fourier mask (boolean)
        
        Returns
        -------
        frame : array
            frame of which bad pixels were corrected
        """
        
        #
        w_ft = np.where(fmask)
        w = np.where(bad_pixels)
        
        # Maps the bad pixels onto the zero region of the Fourier plane
        bad_mat = np.zeros((len(w[0]), len(w_ft[0])*2))
        
        #
        x_half = int(frame.shape[0]/2.)
        y_half = int(frame.shape[1]/2.)
        xy = np.meshgrid(2.*np.pi*np.arange(y_half+1)/float(frame.shape[1]),\
                         2.*np.pi*(((np.arange(frame.shape[0])+x_half) % frame.shape[0])-x_half)/float(frame.shape[0]))
        
        #
        for i in range(0, len(w[0])):
            bad_pixels_ft = np.exp(-1j*(w[0][i]*xy[1]+w[1][i]*xy[0]))
            bad_mat[i, :] = np.append(bad_pixels_ft[w_ft].real, bad_pixels_ft[w_ft].imag)
        
        # Equation 19 of Ireland 2013
        dummy = np.transpose(np.conj(bad_mat))
        inverse_bad_mat = np.dot(dummy, np.linalg.inv(np.dot(bad_mat, dummy))) # FIX
        
        #
        frame[w] = 0.
        frame_ft = (np.fft.rfft2(frame))[w_ft]
        
        #
        corr = -np.real(np.dot(np.append(frame_ft.real, frame_ft.imag), inverse_bad_mat))
        frame[w] += corr
        return frame


class dark():
    """
    Make darks
    """
    
    
    def __init__(self,
                 ddir,
                 rdir,
                 dark_paths):
        """
        Make darks
        
        Parameters
        ----------
        ddir : str
            path of data directory
        rdir : str
            path of reduction directory
        dark_paths : list
            list of dark paths
        """
        
        # Path of data directory
        self.ddir = ddir
        
        # Path of reduction directory
        self.rdir = rdir
        
        # Get relevant dark properties
        dark_props = []
        for i in range(0, len(dark_paths)):
            try:
                fits_file = pyfits.open(self.ddir+dark_paths[i]+'.fits')
            except:
                try:
                    dark_paths[i] = dark_paths[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+dark_paths[i]+'.fits')
                except:
                    raise UserWarning('Could not find dark file '+dark_paths[i])
            fits_header = fits_file[0].header
            dark_props += [(fits_header['NAXIS2'],                     # x-axis
                            fits_header['NAXIS1'],                     # y-axis
                            fits_header['EXPTIME'],                    # exposure time (seconds)
                            fits_header['HIERARCH ESO INS CWLEN'],     # filter wavelength (microns)
                            fits_header['HIERARCH ESO INS OPTI6 ID'],  # filter name
                            fits_header['HIERARCH ESO INS PIXSCALE'])] # pixel scale (arcseconds)
        fits_file.close()
        
        # Make master dark for each unique set of properties
        used_props = []
        for i in range(0, len(dark_props)):
            if (dark_props[i] not in used_props):
                dark_list = []
                for j in range(0, len(dark_props)):
                    if (dark_props[i] == dark_props[j]):
                        dark_list += [dark_paths[j]]
                self._make_dark(dark_list=dark_list)
                used_props += [dark_props[i]]
        
        # Return nothing
        return None
    
    
    def _make_dark(self,
                   dark_list):
        """
        Make master dark for each unique set of properties
        
        Parameters
        ----------
        dark_list : list
            list of dark paths having unique set of properties
        """
        
        # Get darks
        for i in range(0, len(dark_list)):
            try:
                fits_file = pyfits.open(self.ddir+dark_list[i]+'.fits')
            except:
                try:
                    dark_list[i] = dark_list[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+dark_list[i]+'.fits')
                except:
                    raise UserWarning('Could not find dark file '+dark_list[i])
            array = fits_file[0].data
            if (len(array.shape) > 2):
                if (i == 0):
                    darks = array
                else:
                    darks = np.append(darks, array, axis=0)
            else:
                if (i == 0):
                    darks = np.zeros((1, array.shape[0], array.shape[1]))
                    darks[0, :, :] = array
                else:
                    dummy = np.zeros((1, array.shape[0], array.shape[1]))
                    dummy[0, :, :] = array
                    darks = np.append(darks, dummy, axis=0)
            
            
            # Get header
            if (i == 0):
                fits_header = fits_file[0].header
        fits_file.close()
        print('Loaded darks of shape '+str(darks.shape))
        
        # Get median, maximum and variance
        med_dark = np.median(darks, axis=0)
        print('Filtering with median filter of size 9')
        med_dark_filtered = nd.median_filter(med_dark, size=9)
        
        max_dark = np.max(darks, axis=0)
        var_dark = np.zeros((darks.shape[1], darks.shape[2]))
        for i in range(0, darks.shape[0]):
            var_dark += (darks[i, :, :]-med_dark)**2.
        var_dark -= (max_dark-med_dark)**2. # Throw away maximum
        var_dark /= (darks.shape[0]-2.)
        
        # Find pixels with bad median or bad variance
        med_diff = np.maximum(np.median(np.abs(med_dark-med_dark_filtered)), 1.)
        print('Median difference: '+str(med_diff))
        med_var_diff = np.median(np.abs(var_dark-np.median(var_dark)))
        print('Median variance difference: '+str(med_var_diff))
#        bad_med = np.abs(med_dark-np.median(med_dark)) > 15.*med_diff # USER
        bad_med = np.abs(med_dark-med_dark_filtered) > 10.*med_diff    # for CONICA
#        bad_var = var_dark > np.median(var_dark)+10.*med_var_diff     # USER
        bad_var = var_dark > np.median(var_dark)+100.*med_var_diff     # for CONICA
        print('Pixels with bad median: '+str(np.sum(bad_med)))
        print('Pixels with bad variance: '+str(np.sum(bad_var)))
        bad_pixels = np.logical_or(bad_med, bad_var)
        
#        for i in range(0, 3):
#            print('Replacing bad pixels with median filter of size 5')
#            med_dark[bad_pixels] = nd.median_filter(med_dark, size=5)[bad_pixels]
        
        if (make_plots == True):
            plt.figure()
            plt.imshow(med_dark, vmin=np.median(med_dark[bad_pixels == 0])-3*np.std(med_dark[bad_pixels == 0]), vmax=np.median(med_dark[bad_pixels == 0])+3*np.std(med_dark[bad_pixels == 0]))
            plt.show(block=block_plots)
            plt.close()
        
        out_name = 'dark_bad_pixels.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'a_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(bad_pixels)
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        # Save master dark
        out_name = 'master_dark_'+\
                   str(med_dark.shape[0])+'_'+\
                   str(med_dark.shape[1])+'_'+\
                   str(fits_header['EXPTIME'])[0:4]+'_'+\
                   str(fits_header['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                   str(fits_header['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                   str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
        out_fits = pyfits.HDUList()
        out_fits.append(pyfits.ImageHDU(med_dark, fits_header))
        out_fits.append(pyfits.ImageHDU(np.uint8(bad_pixels)))
        out_fits.writeto(self.rdir+out_name+'.fits',\
                         output_verify='ignore',\
                         overwrite=True)
        
        # Return nothing
        return None


class flat():
    """
    Make flats
    """
    
    
    def __init__(self,
                 ddir,
                 rdir,
                 flat_paths):
        """
        Make flats
        
        Parameters
        ----------
        ddir : str
            path of data directory
        rdir : str
            path of reduction directory
        flat_paths : list
            list of flat paths
        """
        
        # Path of data directory
        self.ddir = ddir
        
        # Path of reduction directory
        self.rdir = rdir
        
        # Get relevant flat properties
        flat_props = []
        for i in range(0, len(flat_paths)):
            try:
                fits_file = pyfits.open(self.ddir+flat_paths[i]+'.fits')
            except:
                try:
                    flat_paths[i] = flat_paths[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+flat_paths[i]+'.fits')
                except:
                    raise UserWarning('Could not find flat file '+flat_paths[i])
            fits_header = fits_file[0].header
            flat_props += [(fits_header['NAXIS2'],                     # x-axis
                            fits_header['NAXIS1'],                     # y-axis
                            fits_header['EXPTIME'],                    # exposure time (seconds)
                            fits_header['HIERARCH ESO INS CWLEN'],     # filter wavelength (microns)
                            fits_header['HIERARCH ESO INS OPTI6 ID'],  # filter name
                            fits_header['HIERARCH ESO INS PIXSCALE'])] # pixel scale (arcseconds)
        fits_file.close()
        
        # Make master flat for each unique set of properties
        used_props = []
        for i in range(0, len(flat_props)):
            if (flat_props[i] not in used_props):
                flat_list = []
                for j in range(0, len(flat_props)):
                    if (flat_props[i] == flat_props[j]):
                        flat_list += [flat_paths[j]]
                self._make_flat(flat_list=flat_list)
                used_props += [flat_props[i]]
        
        # Return nothing
        return None
    
    
    def _make_flat(self,
                   flat_list):
        """
        Make master flat for each unique set of properties
        
        Parameters
        ----------
        flat_list : list
            list of flat paths having unique set of properties
        """
        
        # Get flats
        for i in range(0, len(flat_list)):
            try:
                fits_file = pyfits.open(self.ddir+flat_list[i]+'.fits')
            except:
                try:
                    flat_list[i] = flat_list[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+flat_list[i]+'.fits')
                except:
                    raise UserWarning('Could not find flat file '+flat_list[i])
            array = fits_file[0].data
            if (len(array.shape) > 2):
                if (i == 0):
                    flats = array
                else:
                    flats = np.append(flats, array, axis=0)
            else:
                if (i == 0):
                    flats = np.zeros((1, array.shape[0], array.shape[1]))
                    flats[0, :, :] = array
                else:
                    dummy = np.zeros((1, array.shape[0], array.shape[1]))
                    dummy[0, :, :] = array
                    flats = np.append(flats, dummy, axis=0)
            
            
            # Get header
            if (i == 0):
                fits_header = fits_file[0].header
        fits_file.close()
        print('Loaded flats of shape '+str(flats.shape))
        
        # Get median, maximum and variance
        med_flat = np.median(flats, axis=0)
        print('Filtering with median filter of size 9')
        med_flat_filtered = nd.median_filter(med_flat, size=9)
        
        max_flat = np.max(flats, axis=0)
        var_flat = np.zeros((flats.shape[1], flats.shape[2]))
        for i in range(0, flats.shape[0]):
            var_flat += (flats[i, :, :]-med_flat)**2.
        var_flat -= (max_flat-med_flat)**2. # Throw away maximum
        var_flat /= (flats.shape[0]-2.)
        
        # Find pixels with bad median or bad variance
        med_diff = np.maximum(np.median(np.abs(med_flat-med_flat_filtered)), 1.)
        print('Median difference: '+str(med_diff))
        med_var_diff = np.median(np.abs(var_flat-np.median(var_flat)))
        print('Median variance difference: '+str(med_var_diff))
#        bad_med = np.abs(med_flat-np.median(med_flat)) > 15.*med_diff # USER
        bad_med = np.abs(med_flat-med_flat_filtered) > 7.*med_diff     # for CONICA
#        bad_var = var_flat > np.median(var_flat)+10.*med_var_diff     # USER
        bad_var = var_flat > np.median(var_flat)+10.*med_var_diff      # for CONICA
        print('Pixels with bad median: '+str(np.sum(bad_med)))
        print('Pixels with bad variance: '+str(np.sum(bad_var)))
        bad_pixels = np.logical_or(bad_med, bad_var)
        
#        for i in range(0, 3):
#            print('Replacing bad pixels with median filter of size 9')
#            med_flat[bad_pixels] = nd.median_filter(med_flat, size=9)[bad_pixels]
        
        # Find appropriate master dark
        dark_path = self._get_dark(fits_header)
        
        # Copies to be used with darks of different shape
        static_med_flat = med_flat.copy()
        static_bad_pixels = bad_pixels.copy()
        
        # Subtract dark and normalize flat
        med_flat = med_flat-pyfits.getdata(self.rdir+dark_path, 0)
        bad_pixels = np.logical_or(bad_pixels, pyfits.getdata(self.rdir+dark_path, 1) > 0)
#        med_flat[np.where(bad_pixels)] = np.median(med_flat[bad_pixels == 0])
        med_flat /= np.median(med_flat)
        med_flat[med_flat == 0] = 1 # DODGY: Remove stripes from median flat
        
        if (make_plots == True):
            plt.figure()
            plt.imshow(med_flat, vmin=np.median(med_flat[bad_pixels == 0])-3*np.std(med_flat[bad_pixels == 0]), vmax=np.median(med_flat[bad_pixels == 0])+3*np.std(med_flat[bad_pixels == 0]))
            plt.show(block=block_plots)
            plt.close()
        
        out_name = 'flat_bad_pixels.png'
        same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
        out_name = 'b_%03d' % len(same_tar)+'_'+out_name
        plt.figure()
        plt.imshow(bad_pixels)
        plt.savefig(self.rdir+out_name)
        if (make_plots == True):
            plt.show(block=block_plots)
        plt.close()
        
        # Save master flat
        out_name = 'master_flat_'+\
                   str(med_flat.shape[0])+'_'+\
                   str(med_flat.shape[1])+'_'+\
                   str(fits_header['EXPTIME'])[0:4]+'_'+\
                   str(fits_header['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                   str(fits_header['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                   str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
        out_fits = pyfits.HDUList()
        out_fits.append(pyfits.ImageHDU(med_flat, fits_header))
        out_fits.append(pyfits.ImageHDU(np.uint8(bad_pixels)))
        out_fits.writeto(self.rdir+out_name+'.fits',\
                         output_verify='ignore',\
                         overwrite=True)
        
        print('Looking for master darks of different shape')
        
        # Get all master darks
        master_darks = [f for f in os.listdir(self.rdir) if f.startswith('master_dark')]
        
        # Find master darks of different shape
        for i in range(0, len(master_darks)):
            if ((str(fits_header['NAXIS2']) not in master_darks[i] or\
                str(fits_header['NAXIS1']) not in master_darks[i]) and\
                str(fits_header['EXPTIME'])[0:4] in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                dark_path = master_darks[i]
                dark_header = pyfits.getheader(self.rdir+dark_path)
                print('Found master dark of shape '+str(pyfits.getdata(self.rdir+dark_path, 0).shape))
                if (fits_header['NAXIS1'] % 2 != 0 or\
                    fits_header['NAXIS2'] % 2 != 0 or\
                    dark_header['NAXIS1'] % 2 != 0 or\
                    dark_header['NAXIS2'] % 2 != 0):
                    raise UserWarning('Cropping requires arrays of even shape')
                elif (fits_header['NAXIS1'] < dark_header['NAXIS1'] or\
                    fits_header['NAXIS2'] < dark_header['NAXIS2']):
                    print('Skipping dark because it is too small')
                else:
                    print('Cropping master flat and processing it again')
                    
                    # Crop master flat
                    x_min = int(static_med_flat.shape[0]/2.)-int(float(dark_header['NAXIS2'])/2.)
                    y_min = int(static_med_flat.shape[1]/2.)-int(float(dark_header['NAXIS1'])/2.)
                    med_flat = static_med_flat[x_min:x_min+int(dark_header['NAXIS2']), y_min:y_min+int(dark_header['NAXIS1'])]
                    bad_pixels = static_bad_pixels[x_min:x_min+int(dark_header['NAXIS2']), y_min:y_min+int(dark_header['NAXIS1'])]
                    
                    if (make_plots == True):
                        plt.figure()
                        plt.imshow(bad_pixels+pyfits.getdata(self.rdir+dark_path, 1))
                        plt.colorbar()
                        plt.show(block=block_plots)
                    
                    # Subtract dark and normalize flat
                    med_flat = med_flat-pyfits.getdata(self.rdir+dark_path, 0)
                    bad_pixels = np.logical_or(bad_pixels, pyfits.getdata(self.rdir+dark_path, 1) > 0)
#                    med_flat[np.where(bad_pixels)] = np.median(med_flat)
                    med_flat /= np.median(med_flat)
                    med_flat[med_flat == 0] = 1 # DODGY: Remove stripes from median flat
                    
                    if (make_plots == True):
                        plt.figure()
                        plt.imshow(med_flat, vmin=np.median(med_flat[bad_pixels == 0])-3*np.std(med_flat[bad_pixels == 0]), vmax=np.median(med_flat[bad_pixels == 0])+3*np.std(med_flat[bad_pixels == 0]))
                        plt.show(block=block_plots)
                    
                    out_name = 'flat_bad_pixels.png'
                    same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                    out_name = 'b_%03d' % len(same_tar)+'_'+out_name
                    plt.figure()
                    plt.imshow(bad_pixels)
                    plt.savefig(self.rdir+out_name)
                    if (make_plots == True):
                        plt.show(block=block_plots)
                    plt.close()
                    
                    # Save master flat
                    dummy_header = fits_header.copy()
                    dummy_header['NAXIS1'] = dark_header['NAXIS1']
                    dummy_header['NAXIS2'] = dark_header['NAXIS2']
                    out_name = 'master_flat_'+\
                               str(med_flat.shape[0])+'_'+\
                               str(med_flat.shape[1])+'_'+\
                               str(fits_header['EXPTIME'])[0:4]+'_'+\
                               str(fits_header['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                               str(fits_header['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                               str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
                    out_fits = pyfits.HDUList()
                    out_fits.append(pyfits.ImageHDU(med_flat, dummy_header))
                    out_fits.append(pyfits.ImageHDU(np.uint8(bad_pixels)))
                    out_fits.writeto(self.rdir+out_name+'.fits',\
                                     output_verify='ignore',\
                                     overwrite=True)
            elif ((str(fits_header['NAXIS2']) not in master_darks[i] or\
                str(fits_header['NAXIS1']) not in master_darks[i]) and\
                str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                dark_path = master_darks[i]
                dark_header = pyfits.getheader(self.rdir+dark_path)
                print('Found master dark with different exposure time of shape '+str(pyfits.getdata(self.rdir+dark_path, 0).shape))
                if (fits_header['NAXIS1'] % 2 != 0 or\
                    fits_header['NAXIS2'] % 2 != 0 or\
                    dark_header['NAXIS1'] % 2 != 0 or\
                    dark_header['NAXIS2'] % 2 != 0):
                    raise UserWarning('Cropping requires arrays of even shape')
                elif (fits_header['NAXIS1'] < dark_header['NAXIS1'] or\
                    fits_header['NAXIS2'] < dark_header['NAXIS2']):
                    print('Skipping dark because it is too small')
                else:
                    print('Cropping master flat and processing it again')
                    
                    # Crop master flat
                    x_min = int(static_med_flat.shape[0]/2.)-int(float(dark_header['NAXIS2'])/2.)
                    y_min = int(static_med_flat.shape[1]/2.)-int(float(dark_header['NAXIS1'])/2.)
                    med_flat = static_med_flat[x_min:x_min+int(dark_header['NAXIS2']), y_min:y_min+int(dark_header['NAXIS1'])]
                    bad_pixels = static_bad_pixels[x_min:x_min+int(dark_header['NAXIS2']), y_min:y_min+int(dark_header['NAXIS1'])]
                    
                    if (make_plots == True):
                        plt.figure()
                        plt.imshow(bad_pixels+pyfits.getdata(self.rdir+dark_path, 1))
                        plt.colorbar()
                        plt.show(block=block_plots)
                    
                    # Subtract dark and normalize flat
                    ratio = float(fits_header['EXPTIME'])/float(dark_header['EXPTIME'])
                    med_flat = med_flat-pyfits.getdata(self.rdir+dark_path, 0)*ratio
                    bad_pixels = np.logical_or(bad_pixels, pyfits.getdata(self.rdir+dark_path, 1) > 0)
#                    med_flat[np.where(bad_pixels)] = np.median(med_flat)
                    med_flat /= np.median(med_flat)
                    med_flat[med_flat == 0] = 1 # DODGY: Remove stripes from median flat
                    
                    if (make_plots == True):
                        plt.figure()
                        plt.imshow(med_flat)
                        plt.show(block=block_plots)
                    
                    out_name = 'flat_bad_pixels.png'
                    same_tar = [f for f in os.listdir(self.rdir) if (out_name in f)]
                    out_name = 'b_%03d' % len(same_tar)+'_'+out_name
                    plt.figure()
                    plt.imshow(bad_pixels)
                    plt.savefig(self.rdir+out_name)
                    if (make_plots == True):
                        plt.show(block=block_plots)
                    plt.close()
                    
                    # Save master flat
                    dummy_header = fits_header.copy()
                    dummy_header['NAXIS1'] = dark_header['NAXIS1']
                    dummy_header['NAXIS2'] = dark_header['NAXIS2']
                    out_name = 'master_flat_'+\
                               str(med_flat.shape[0])+'_'+\
                               str(med_flat.shape[1])+'_'+\
                               str(fits_header['EXPTIME'])[0:4]+'_'+\
                               str(fits_header['HIERARCH ESO INS CWLEN'])[0:4]+'_'+\
                               str(fits_header['HIERARCH ESO INS OPTI6 ID'])+'_'+\
                               str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2]
                    out_fits = pyfits.HDUList()
                    out_fits.append(pyfits.ImageHDU(med_flat, dummy_header))
                    out_fits.append(pyfits.ImageHDU(np.uint8(bad_pixels)))
                    out_fits.writeto(self.rdir+out_name+'.fits',\
                                     output_verify='ignore',\
                                     overwrite=True)
        
        # Return nothing
        return None
    
    
    def _get_dark(self,
                  fits_header):
        """
        Find appropriate master dark
        
        Parameters
        ----------
        fits_header : header
            header of master flat for which appropriate master dark should be found
        
        Returns
        -------
        dark_path : str
            path of appropriate master dark
        """
        
        # Get all master darks
        master_darks = [f for f in os.listdir(self.rdir) if f.startswith('master_dark')]
        
        # Find appropriate master dark
        flag = False
        for i in range(0, len(master_darks)):
            if (str(fits_header['NAXIS2']) in master_darks[i] and\
                str(fits_header['NAXIS1']) in master_darks[i] and\
                str(fits_header['EXPTIME'])[0:4] in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS CWLEN'])[0:4] in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS OPTI6 ID']) in master_darks[i] and\
                str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                dark_path = master_darks[i]
                flag = True
                print('Found appropriate master dark')
                break
        if (flag == False):
            print('Searching among different wavelengths and filters')
            for i in range(0, len(master_darks)):
                if (str(fits_header['NAXIS2']) in master_darks[i] and\
                    str(fits_header['NAXIS1']) in master_darks[i] and\
                    str(fits_header['EXPTIME'])[0:4] in master_darks[i] and\
                    str(fits_header['HIERARCH ESO INS PIXSCALE']*1000.)[0:2] in master_darks[i]):
                    dark_path = master_darks[i]
                    flag = True
                    print('Found appropriate master dark')
                    break
        if (flag == False):
            raise UserWarning('Could not find appropriate master dark')
        
        # Return appropriate master dark
        return dark_path
