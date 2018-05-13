"""
PyConica, a Python package to reduce NACO data. Information about NACO can be
found at https://www.eso.org/sci/facilities/paranal/instruments/naco.html. This
library is maintained on GitHub at https://github.com/kammerje/PyConica.

Author: Jens Kammerer
Version: 1.0.1
Last edited: 12.05.18
"""


# PREAMBLE
#==============================================================================
import sys
sys.path.append('/home/kjens/Python/Packages/opticstools/opticstools/opticstools')

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import opticstools as ot
import os
import scipy.ndimage as nd
from scipy.optimize import minimize
from xml.dom import minidom


# PARAMETERS
#==============================================================================
make_plots = True
make_unimportant_plots = False
block_plots = False

saturation_threshold = 16400. # see https://www.eso.org/observing/dfo/quality/NACO/qc/detmon_qc1.html#top
saturation_threshold = 16000. # Real value based on detcheck
linearity_range = 8500. # see https://www.eso.org/observing/dfo/quality/NACO/qc/detmon_qc1.html#top
linearize = True
detcheck_dir = '/priv/mulga2/kjens/NACO/girard_2015/detcheck/data_with_raw_calibs/'

shift_as_prior = True
sub_size = 96

skip_processed_data = True


# MAIN
#==============================================================================
def get_b(c, d):
    """
    Cubic polynomial h which is fit to correction curve with boundary conditions h(linearity_range) = linearity_range and h'(linearity_range) = 1
    """
    return 1.-2.*c*linearity_range-3.*d*linearity_range**2

def get_a(b, c, d):
    """
    Cubic polynomial h which is fit to correction curve with boundary conditions h(linearity_range) = linearity_range and h'(linearity_range) = 1
    """
    return linearity_range-b*linearity_range-c*linearity_range**2-d*linearity_range**3

def fit(x, x_in, y_in):
    """
    Cubic polynomial h which is fit to correction curve with boundary conditions h(linearity_range) = linearity_range and h'(linearity_range) = 1
    """
    d = x[0]
    c = x[1]
    b = get_b(c=c,
              d=d)
    a = get_a(b=b,
              c=c,
              d=d)
    
    res = np.zeros(x_in.shape)
    for i in range(x_in.shape[0]):
        res[i] = a+b*x_in[i]+c*x_in[i]**2+d*x_in[i]**3-y_in[i]
    
    return np.sum(np.abs(res))

def fit_polynomial_to_detcheck(detcheck_dir):
    """
    Parameters
    ----------
    detcheck_dir : str
        Path of detcheck directory
    
    Returns
    -------
    pp : array
        Array with coefficients of cubic polynomial which is fit to correction curve
    """
    
    # Identify detcheck files
    detcheck_files = [f for f in os.listdir(detcheck_dir) if f.endswith('.fits')]
    
    # Open detcheck files
    exptime = []
    for i in range(len(detcheck_files)):
        try:
            fits_file = pyfits.open(detcheck_dir+detcheck_files[i])
        except:
            try:
                detcheck_files[i] = detcheck_files[i].replace(':', '_')
                fits_file = pyfits.open(detcheck_dir+detcheck_files[i])
            except:
                raise UserWarning('Could not find detcheck file '+detcheck_files[i])
        
        # Read header
        fits_header = fits_file[0].header
        if (fits_header['HIERARCH ESO TPL ID'] != 'NACO_img_cal_Linearity'):
            raise UserWarning('DETCHECK files must be of type NACO_img_cal_Linearity')
        
        # Read data into array
        detcheck_temp = fits_file[0].data
        fits_file.close()
        if (len(detcheck_temp.shape) == 3):
            if (i == 0):
                detcheck = detcheck_temp
            else:
                detcheck = np.append(detcheck, detcheck_temp, axis=0)
            exptime += [fits_header['EXPTIME']]*detcheck_temp.shape[0]
                
        else:
            if (i == 0):
                detcheck = np.zeros((1, detcheck_temp.shape[0], detcheck_temp.shape[1]))
                detcheck[0] = detcheck_temp
            else:
                dummy = np.zeros((1, detcheck_temp.shape[0], detcheck_temp.shape[1]))
                dummy[0] = detcheck_temp
                detcheck = np.append(detcheck, dummy, axis=0)
            exptime += [fits_header['EXPTIME']]
    print('Read detcheck files of shape '+str(detcheck.shape))
    
    # Remove stripes from broken quadrant
    mask1 = detcheck < 32768.
    
    # Compute median of detcheck files with similar exposure time
    mask2 = np.zeros(mask1.shape)
    x = [] # Exposure time
    y = [] # Median ADUs
    for i in np.unique(exptime):
        mask2[:, :, :] = 0
        mask2[np.where(np.array(exptime) == i)[0]] = 1
        mask = mask1 & mask2.astype('bool')
        x += [i]
        y += [np.median(detcheck[mask])]
    
    # Fit linear and cubic polynomial to y(x)
    x = np.array(x)
    y = np.array(y)
    ppL = np.polyfit(x[y < linearity_range], y[y < linearity_range], 1)
    yyL = ppL[1]+ppL[0]*x
    ppC = np.polyfit(x[y < saturation_threshold], y[y < saturation_threshold], 3)
    yyC = ppC[3]+ppC[2]*x+ppC[1]*x**2+ppC[0]*x**3
    
    # Fit cubic polynomial to correction curve between linearity_range and saturation_threshold
    mask1 = y < saturation_threshold
    mask2 = y > linearity_range
    mask = mask1 & mask2
    
    x0 = np.array([0, 0])
    pp = minimize(fun=fit, # Cubic polynomial with smooth transition to identity at linearity_range
                  x0=x0,
                  args=(yyC[mask], yyL[mask]),
                  method='Nelder-Mead')
    d = pp.x[0]
    c = pp.x[1]
    b = get_b(c=c,
              d=d)
    a = get_a(b=b,
              c=c,
              d=d)
    pp = np.array([d, c, b, a])
    xx = np.linspace(linearity_range, saturation_threshold, 100)
    yy = pp[3]+pp[2]*xx+pp[1]*xx**2+pp[0]*xx**3
    
    if (make_plots):
        f, axarr = plt.subplots(1, 2, figsize=(12, 9))
        axarr[0].plot(x, y, 'o-', label='detcheck data')
        axarr[0].plot(x, yyL, 'o-', label='linear fit')
        axarr[0].plot(x, yyC, 'o-', label='cubic fit')
        axarr[0].axhline(saturation_threshold, color='red', label='saturation threshold')
        axarr[0].set_xlabel('Exposure time')
        axarr[0].set_ylabel('ADU (measured)')
        axarr[0].legend()
        axarr[1].plot(yyC, yyL, 'o-', label='correction curve')
        axarr[1].plot(xx, yy, '-', label='cubic fit')
        axarr[1].axvline(linearity_range, color='red', ls='--', label='linearity range')
        axarr[1].axvline(saturation_threshold, color='red', label='saturation threshold')
        axarr[1].set_xlabel('ADU (measured)')
        axarr[1].set_ylabel('ADU (corrected)')
        axarr[1].legend()
        plt.suptitle('Detector linearity correction')
        plt.savefig(detcheck_dir+'detector_linearity_correction.pdf', bbox_inches='tight')
        plt.show(block=block_plots)
        plt.close()
    
    # Return array with coefficients of cubic polynomial which is fit to correction curve
    return pp

print('--> Computing detector linearity correction')
pp = fit_polynomial_to_detcheck(detcheck_dir=detcheck_dir)
if (linearize):
    print('Pixels with ADUs between %.0f and %.0f will be linearized' % (linearity_range, saturation_threshold))

def linearize(frame, pp):
    """
    Parameters
    ----------
    frame : array
        Frame to be linearized
    pp : array
        Array with coefficients of cubic polynomial which is fit to correction curve
    
    Returns
    -------
    frame_linearized : array
        Linearized frame
    """
    
    # Linearize all pixels with counts between linearity_range and saturation_threshold
    frame_linearized = frame.copy()
    mask1 = frame < saturation_threshold
    mask2 = frame > linearity_range
    mask = mask1 & mask2
    frame_linearized[mask] = pp[3]+pp[2]*frame[mask]+pp[1]*frame[mask]**2+pp[0]*frame[mask]**3
    
    # Return linearized frame
    return frame_linearized

if (make_unimportant_plots):
    identity = np.linspace(-5000, saturation_threshold, 1000)
    plt.figure(figsize=(12, 9))
    plt.plot(identity, identity, label='id')
    plt.plot(identity, linearize(identity, pp), label='corrected')
    plt.xlabel('ADU (measured)')
    plt.ylabel('ADU (corrected)')
    plt.legend()
    plt.title('Correction curve')
    plt.show(block=block_plots)
    plt.close()

class cube(object):
    
    def __init__(self,
                 ddir,
                 rdir,
                 cdir):
        """
        Parameters
        ----------
        ddir : str
            Path of data directory
        rdir : str
            Path of reduction directory
        cdir : str
            Path of cube directory
        """
        
        print('--> Initializing directories')
        self.ddir = ddir
        self.rdir = rdir
        self.cdir = cdir
        
        pass
    
    def process_ob(self,
                   xml_path):
        """
        Parameters
        ----------
        xml_path : str
            Path of xml file (must be sub-directory of ddir)
        """
        
        print('--> Reading dark, flat and data paths from xml file')
        dark_paths, flat_paths, data_paths = self.__read_from_xml(xml_path=xml_path)
        print('--------------------')
        
#        return pyfits.getheader(self.ddir+data_paths[0]+'.fits')['HIERARCH ESO TPL START '][:11]
        
        print('--> Making master darks')
        if (len(dark_paths) != 0):
            dark(ddir=self.ddir,
                 rdir=self.rdir,
                 dark_paths=dark_paths)
        print('--------------------')
        
        print('--> Making master flats')
        if (len(flat_paths) != 0):
            flat(ddir=self.ddir,
                 rdir=self.rdir,
                 dark_paths=dark_paths,
                 flat_paths=flat_paths)
        print('--------------------')
        
        print('--> Verifying data cubes')
        data_paths = self.__verify(data_paths=data_paths)
        print('--> Cleaning data cubes')
        cubes = []
        for i in range(len(data_paths)):
            cubes += [self.__clean(data_path=data_paths[i],
                         dark_paths=dark_paths,
                         flat_paths=flat_paths)]
        print('--> Background subtracting data cubes')
        self.__jitter_subtraction(cubes=cubes)
        print('--> Bad pixel correcting data cubes')
        self.__bad_pixel_correction(cubes=cubes)
        print('--------------------')
    
    def __read_from_xml(self,
                        xml_path):
        """
        Parameters
        ----------
        xml_path : str
            Path of xml file (must be sub-directory of ddir)
        
        Returns
        -------
        dark_paths : list
            List of dark paths
        flat_paths : list
            List of flat paths
        data_paths : list
            List of data paths
        """
        
        # Open xml file
        xml_file = minidom.parse(self.ddir+xml_path)
        
        # Get all items of type file
        xml_items = xml_file.getElementsByTagName('file')
        
        # Extract dark, flat and data paths
        dark_paths = []
        flat_paths = []
        data_paths = []
        for item in xml_items:
            if (item.attributes['category'].value == 'CAL_DARK'):
                dark_paths += [item.attributes['name'].value]
            elif (item.attributes['category'].value == 'CAL_FLAT_TW'):
                flat_paths += [item.attributes['name'].value]
            elif (item.attributes['category'].value == 'IM_JITTER_OBJ'):
                data_paths += [item.attributes['name'].value]
            else:
                print('Could not match '+item.attributes['category'].value)
        
        # Remove duplicates
        dark_paths = list(set(dark_paths))
        flat_paths = list(set(flat_paths))
        data_paths = list(set(data_paths))
        
        print(str(len(dark_paths))+' dark files identified')
        print(str(len(flat_paths))+' flat files identified')
        print(str(len(data_paths))+' data files identified')
        
        logfile = open('log.txt', 'a')
        logfile.write(str(len(dark_paths))+' dark files identified\n')
        logfile.write(str(len(flat_paths))+' flat files identified\n')
        logfile.write(str(len(data_paths))+' data files identified\n')
        logfile.close()
        
        # Return dark, flat and data paths
        return dark_paths, flat_paths, data_paths
    
    def __verify(self,
                 data_paths):
        """
        Parameters
        ----------
        data_paths : list
            List of data paths
        
        Returns
        -------
        data_paths_ver : list
            List of verified data paths
        """
        
        # Reject data cubes with less than 10 frames
        data_paths_ver = []
        for i in range(len(data_paths)):
            try:
                fits_file = pyfits.open(self.ddir+data_paths[i]+'.fits')
                fits_header = fits_file[0].header
                fits_file.close()
            except:
                try:
                    data_paths[i] = data_paths[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+data_paths[i]+'.fits')
                    fits_header = fits_file[0].header
                    fits_file.close()
                except:
                    raise UserWarning('Could not find data cube '+data_paths[i])
            if (int(fits_header['NAXIS3']) >= 10):
                data_paths_ver += [data_paths[i]]
        print('Rejected '+str(len(data_paths)-len(data_paths_ver))+' data cubes due to low frame number')
        
        logfile = open('log.txt', 'a')
        logfile.write('Rejected '+str(len(data_paths)-len(data_paths_ver))+' data cubes due to low frame number\n')
        logfile.close()
        
        # Return list of verified data paths
        return data_paths_ver
    
    def __find_master_dark(self,
                           props,
                           dark_paths):
        """
        Parameters
        ----------
        props : dict
            Dictionary of properties
        dark_paths : list
            List of dark paths
        
        Returns
        -------
        dark_path : str
            Path of appropriate master dark
        """
        
        # Find path of appropriate master dark
        dark_path = [f for f in os.listdir(self.rdir) if ('master_dark' in f) and (str(props['NAXIS2']) in f) and (str(props['NAXIS1']) in f) and ('.fits' in f)]
        if (len(dark_path) > 1):
            print('Identified more than 1 appropriate master darks, using first one')
        
        # Return path of appropriate master dark
        return dark_path[0]
    
    def __find_master_flat(self,
                           props,
                           flat_paths):
        """
        Parameters
        ----------
        props : dict
            Dictionary of properties
        flat_paths : list
            List of flat paths
        
        Returns
        -------
        flat_path : str
            Path of appropriate master flat
        """
        
        # Find path of appropriate master flat
        flat_path = [f for f in os.listdir(self.rdir) if ('master_flat' in f) and (str(props['NAXIS2']) in f) and (str(props['NAXIS1']) in f) and (str(props['HIERARCH ESO INS CWLEN']) in f) and ('.fits' in f)]
        if (len(flat_path) > 1):
            print('Identified more than 1 appropriate master flats, using first one')
        
        # Return path of appropriate master flat
        return flat_path[0]
    
    def __fix_bad_pixels(self,
                         frame,
                         bad_pixels,
                         fmask):
        """
        Parameters
        ----------
        frame : array
            Frame to be corrected
        bad_pixels : array
            Bad pixel map
        fmask : array
            Fourier mask
        
        Returns
        -------
        frame : array
            Corrected frame
        """
        
        #
        w_ft = np.where(fmask)
        w = np.where(bad_pixels)
        
        # Maps the bad pixels onto the zero region of the Fourier plane
        bad_mat = np.zeros((len(w[0]), len(w_ft[0])*2))
        
        #
        x_half = int(frame.shape[0]/2)
        y_half = int(frame.shape[1]/2)
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
    
    def __pmask(self,
                fits_header):
        """
        Parameters
        ----------
        fits_header : header
            Header of data cube for which pupil mask should be made
        
        Returns
        -------
        pmask : array
            Pupil mask
        """
        
        # Set telescope dimensions
        mirror_size = 8.2 # meters
        
        # Extract relevant information from header
        pscale = float(fits_header['PSCALE'])
        cwave = float(fits_header['CWAVE'])
        
        # Set scale ratio for Fourier transformation
        ratio = cwave/(pscale/1000./60./60.*np.pi/180.*sub_size)
        
        # Make pupil mask
        pmask = ot.circle(sub_size, mirror_size/ratio)
        
        # Return pupil mask
        return pmask
    
    def __fmask(self,
                pmask):
        """
        Parameters
        ----------
        pmask : array
            Pupil mask
        
        Returns
        -------
        fmask : array
            Fourier mask
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
        
        # Return Fourier mask
        return fmask
    
    def __clean(self,
                data_path,
                dark_paths,
                flat_paths):
        """
        Parameters
        ----------
        data_path : str
            Path to data cube
        dark_paths : list
            List of dark paths
        flat_paths : list
            List of flat paths
        
        Returns
        -------
        out_name : str
            Output file name
        """
        
        # Open data cube and read header
        try:
            fits_file = pyfits.open(self.ddir+data_path+'.fits')
        except:
            try:
                data_path = data_path.replace(':', '_')
                fits_file = pyfits.open(self.ddir+data_path+'.fits')
            except:
                raise UserWarning('Could not find data cube '+data_path)
        fits_header = fits_file[0].header
        
        # Check if data cube has already been processed
        if (skip_processed_data):
            check = False
            processed_data = [f for f in os.listdir(self.cdir) if (fits_header['HIERARCH ESO OBS TARG NAME'] in f) and ('_cleaned.fits' in f)]
            for i in range(len(processed_data)):
                if (fits_header['ORIGFILE'] == pyfits.getheader(self.cdir+processed_data[i])['ORIGFILE']):
                    check = True
                    break
            if (check):
                print('Skipping data cube '+data_path+', data cube has already been processed')
                return processed_data[i][:-13]
        
        # Read data into array
        data = fits_file[0].data[:-1, :, :] # Last frame is mean
        data = data[1:, :, :] # First frame is rubbish
        fits_file.close()
        print('Read data of shape '+str(data.shape))
        
        # Find saturated pixels
        sat_pixels = data >= saturation_threshold
        
        # Linearize data cube
        print('Linearizing data cube')
        data = linearize(frame=data,
                         pp=pp)
        
        # Stack data cubes with more than 100 frames
        stacked = 1
        if (data.shape[0] > 100):
            print('Stacking data cube')
            stacked = 9
            data_temp = data.copy()
            data = np.zeros((int(data_temp.shape[0]/9), data_temp.shape[1], data_temp.shape[2]))
            sat_pixels_temp = sat_pixels.copy()
            sat_pixels = np.zeros((int(sat_pixels_temp.shape[0]/9), sat_pixels_temp.shape[1], sat_pixels_temp.shape[2]))
            for i in range(int(data_temp.shape[0]/9)):
                data[i] = np.sum(data_temp[i*9:i*9+9], axis=0)
                sat_pixels[i] = np.sum(sat_pixels_temp[i*9:i*9+9], axis=0)
            sat_pixels = sat_pixels > 0.5
            fits_header['EXPTIME'] = fits_header['EXPTIME']*9.
            print('Data cube was stacked to shape '+str(data.shape))
            print('Lost '+str(data_temp.shape[0] % 9)+' files due to stacking')
            
            logfile = open('log.txt', 'a')
            logfile.write('Data cube was stacked to shape '+str(data.shape)+'\n')
            logfile.write('Lost '+str(data_temp.shape[0] % 9)+' files due to stacking\n')
            logfile.close()
            
            del data_temp
            del sat_pixels_temp
        
        # Find appropriate master dark and master flat
        props = {'NAXIS2': fits_header['NAXIS2'], # Size in x-direction
                 'NAXIS1': fits_header['NAXIS1'], # Size in y-direction
                 'EXPTIME': fits_header['EXPTIME'], # Exposure time
                 'HIERARCH ESO INS CWLEN': fits_header['HIERARCH ESO INS CWLEN']} # Filter wavelength
        dark_path = self.__find_master_dark(props=props,
                                            dark_paths=dark_paths)
        flat_path = self.__find_master_flat(props=props,
                                            flat_paths=flat_paths)
        
        # Output file name
        out_name = str(fits_header['HIERARCH ESO OBS TARG NAME'])
        for key in props.keys():
            out_name += '_'+str(props[key])
        index = [f for f in os.listdir(self.cdir) if (out_name in f) and ('.fits' in f)]
        out_name += '_%03d' % len(index)
        
        # Open master dark and master flat
        dark_header = pyfits.getheader(self.rdir+dark_path)
        flat_header = pyfits.getheader(self.rdir+flat_path)
        ratio = float(fits_header['EXPTIME'])/float(dark_header['EXPTIME'])
        dark = pyfits.getdata(self.rdir+dark_path, 0)*ratio
        flat = pyfits.getdata(self.rdir+flat_path, 0)
        bad_pixels = pyfits.getdata(self.rdir+flat_path, 1)
        
        # Extract relevant information from header
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
            gain = 9.8 # from http://www.eso.org/sci/facilities/paranal/instruments/naco/doc/VLT-MAN-ESO-14200-2761_v102.pdf
            readnoise = 4.4 # from http://www.eso.org/sci/facilities/paranal/instruments/naco/doc/VLT-MAN-ESO-14200-2761_v102.pdf
            if (dark_header['RDNOISE'] > readnoise):
                readnoise = dark_header['RDNOISE']
            print('Detector mode is '+mode+', gain = '+str(gain)+', read noise = %.1f' % readnoise)
        else:
            raise UserWarning('Detector mode '+mode+' is not known')
        
        if (make_plots):
            f, axarr = plt.subplots(2, 2, figsize=(12, 9))
            dummy = np.median(data, axis=0)
            p00 = axarr[0, 0].imshow(dummy, vmin=np.median(dummy)-np.std(dummy), vmax=np.median(dummy)+np.std(dummy))
            axarr[0, 0].set_title('Median of raw data cube')
        
        # Clean each frame separately
        pas = np.zeros(data.shape[0])
        bgs = np.zeros(data.shape[0])
        for i in range(data.shape[0]):
            if ((i+1) % 10 == 0):
                print('Cleaning frame '+str(i+1))
            frame = data[i]
            
            # Compute parallactic angle
            # FIXME
            pas[i] = alt+instrument_offset+\
                         (rot_start+(rot_end-rot_start)/float(data.shape[0])*float(i))-\
                         (180.-(parang_start+(parang_end-parang_start)/float(data.shape[0])*float(i)))
            
            # Subtract master dark and normalize by master flat
            frame = np.true_divide(frame-dark, flat)
            
            # Clean bad pixels using median filter of size 5
            frame[bad_pixels == 1] = nd.median_filter(frame, size=5)[bad_pixels == 1]
            
            # Subtract background
            bgs[i] = np.median(frame[bad_pixels == 0])
            frame -= bgs[i]
            
            data[i] = frame
        
        if (make_plots):
            dummy = np.median(data, axis=0)
            p01 = axarr[0, 1].imshow(dummy, vmin=-np.std(dummy), vmax=np.std(dummy))
            axarr[0, 1].set_title('Median of cleaned data cube')
            p10 = axarr[1, 0].imshow(bad_pixels, vmin=0, vmax=1)
            axarr[1, 0].set_title('Bad pixel map')
            p11 = axarr[1, 1].imshow(np.sum(sat_pixels, axis=0)/float(sat_pixels.shape[0]), vmin=0, vmax=1)
            axarr[1, 1].set_title('Saturated pixel map')
            c00 = plt.colorbar(p00, ax=axarr[0, 0])
            c00.set_label('Counts', rotation=270, labelpad=20)
            c01 = plt.colorbar(p01, ax=axarr[0, 1])
            c01.set_label('Counts', rotation=270, labelpad=20)
            c11 = plt.colorbar(p11, ax=axarr[1, 1])
            c11.set_label('Fraction of frames where pixel is saturated', rotation=270, labelpad=20)
            plt.savefig(self.cdir+out_name+'_cleaned.pdf', bbox_inches='tight')
            plt.show(block=block_plots)
            plt.close()
        
        # Shift frames and calculate sum
        shift = np.array([offset_x, offset_y])
        sum_frame = np.zeros((data.shape[1], data.shape[2]))
        for i in range(data.shape[0]):
            sum_frame += np.roll(np.roll(data[i], -shift[0], axis=1), -shift[1], axis=0) # FITS and Python axes are swapped
        
        # Find target coordinates
        if (shift_as_prior):
            x_prior = int(sum_frame.shape[0]/2)
            y_prior = int(sum_frame.shape[1]/2)
            
            # Only search 64 x 64 box around shift
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
        
        # Save data cube
        pri_hdu = pyfits.PrimaryHDU(data)
        pri_hdu.header = fits_header
        pri_hdu.header['EXTNAME'] = 'DATA-CUBE'
        pri_hdu.header['CAMERA'] = 'CONICA'
        pri_hdu.header['DATE-OBS'] = fits_header['MJD-OBS']
        pri_hdu.header['PSCALE'] = float(fits_header['HIERARCH ESO INS PIXSCALE'])*1E3
        pri_hdu.header['CWAVE'] = float(fits_header['HIERARCH ESO INS CWLEN'])*1E-6
        pri_hdu.header['GAIN'] = gain
        pri_hdu.header['RDNOISE'] = readnoise
        pri_hdu.header['STACKED'] = stacked
        pri_hdu.header['MAX_X'] = max_x
        pri_hdu.header['MAX_Y'] = max_y
        pri_hdu.header.add_comment('Data cube')
        pri_hdu.header.add_comment(dark_path)
        pri_hdu.header.add_comment(flat_path)
        
        sec_hdu = pyfits.ImageHDU(np.uint8(bad_pixels))
        sec_hdu.header['EXTNAME']  = 'BAD-PIXEL-MAP'
        sec_hdu.header.add_comment('Bad pixel map')
        
        ter_hdu = pyfits.ImageHDU(np.uint8(sat_pixels))
        ter_hdu.header['EXTNAME']  = 'SAT-PIXEL-MAP'
        ter_hdu.header.add_comment('Saturated pixel map')
        
        pas = pyfits.Column(name='DETPA', format='D', array=pas)
        bgs = pyfits.Column(name='BACKS', format='D', array=bgs)
        tb1_hdu = pyfits.BinTableHDU.from_columns([pas, bgs])
        tb1_hdu.header['EXTNAME']  = 'TEL'
        tb1_hdu.header['TTYPE1']   = ('DETPA', 'Detector position angles (in degrees)')
        tb1_hdu.header['TTYPE2']   = ('BACKS', 'Backgrounds')
        
        out_file = pyfits.HDUList([pri_hdu, sec_hdu, ter_hdu, tb1_hdu])
        out_file.writeto(self.cdir+out_name+'_cleaned.fits', overwrite=True, output_verify='fix')
        
        logfile = open('log.txt', 'a')
        logfile.write('Data cube cleaned and saved as '+out_name+'_cleaned.fits\n')
        logfile.close()
        
        # Return output file name
        return out_name
    
    def __jitter_subtraction(self,
                             cubes):
        """
        Parameters
        ----------
        cubes : list
            List of cube paths
        """
        
        # Extract target coordinates from header
        max_x = []
        max_y = []
        for i in range(len(cubes)):
            fits_header = pyfits.getheader(self.cdir+cubes[i]+'_cleaned.fits')
            max_x += [fits_header['MAX_X']]
            max_y += [fits_header['MAX_Y']]
        
        # Find data cube with furthest target for each data cube
        furthest_cube = []
        for i in range(len(cubes)):
            dist = []
            for j in range(len(cubes)):
                dist += [np.sqrt((max_x[i]-max_x[j])**2+(max_y[i]-max_y[j])**2)]
            furthest_cube += [np.argmax(dist)]
            if ((np.abs(max_x[i]-max_x[furthest_cube[i]]) < sub_size/2) and (np.abs(max_y[i]-max_y[furthest_cube[i]]) < sub_size/2)):
                print('WARNING: Targets might be too close for data cube %.0f!' % (i+1))
        
        for i in range(len(cubes)):
            
            # Open data cube
            print('Processing data cube %.0f' % (i+1))
            fits_header = pyfits.getheader(self.cdir+cubes[i]+'_cleaned.fits', 0)
            data = pyfits.getdata(self.cdir+cubes[i]+'_cleaned.fits', 0)
            bad_pixels = pyfits.getdata(self.cdir+cubes[i]+'_cleaned.fits', 1)
            sat_pixels = pyfits.getdata(self.cdir+cubes[i]+'_cleaned.fits', 2)
            
            # Check if data cube has already been processed
            if (skip_processed_data):
                check = False
                processed_data = [f for f in os.listdir(self.cdir) if (fits_header['HIERARCH ESO OBS TARG NAME'] in f) and ('_jitter.fits' in f)]
                for j in range(len(processed_data)):
                    if (fits_header['ORIGFILE'] == pyfits.getheader(self.cdir+processed_data[j])['ORIGFILE']):
                        check = True
                        break
                if (check):
                    print('Skipping data cube '+cubes[i]+', data cube has already been processed')
                    continue
            
            if (make_plots):
                f, axarr = plt.subplots(2, 2, figsize=(12, 9))
                dummy = np.median(data, axis=0)
                vmin = -np.std(dummy)
                vmax = np.std(dummy)
                dummy = np.roll(np.roll(dummy,\
                                int(sub_size/2)-max_y[i], axis=0),\
                                int(sub_size/2)-max_x[i], axis=1)[0:sub_size, 0:sub_size]
                p00 = axarr[0, 0].imshow(dummy, vmin=vmin, vmax=vmax)
                axarr[0, 0].set_title('Median of cleaned data cube')
            
            # Jitter subtraction and cropping to sub_size
            furthest_data = pyfits.getdata(self.cdir+cubes[furthest_cube[i]]+'_cleaned.fits', 0)
            med_furthest_data = np.median(furthest_data, axis=0)
            sub_data = np.zeros((data.shape[0], sub_size, sub_size))
            sub_bad_pixels = np.zeros((sub_size, sub_size))
            sub_sat_pixels = np.zeros((sat_pixels.shape[0], sub_size, sub_size))
            for j in range(data.shape[0]):
                data[j] -= med_furthest_data
                sub_data[j] = np.roll(np.roll(data[j],\
                                      int(sub_size/2)-max_y[i], axis=0),\
                                      int(sub_size/2)-max_x[i], axis=1)[0:sub_size, 0:sub_size]
                sub_bad_pixels = np.roll(np.roll(bad_pixels,\
                                         int(sub_size/2)-max_y[i], axis=0),\
                                         int(sub_size/2)-max_x[i], axis=1)[0:sub_size, 0:sub_size]
                sub_sat_pixels[j] = np.roll(np.roll(sat_pixels[j],\
                                            int(sub_size/2)-max_y[i], axis=0),\
                                            int(sub_size/2)-max_x[i], axis=1)[0:sub_size, 0:sub_size]
            
            if (make_plots):
                dummy = np.median(sub_data, axis=0)
                p01 = axarr[0, 1].imshow(dummy, vmin=-50, vmax=50)
                axarr[0, 1].set_title('Median of jitter subtracted data cube')
                p10 = axarr[1, 0].imshow(sub_bad_pixels, vmin=0, vmax=1)
                axarr[1, 0].set_title('Bad pixel map')
                p11 = axarr[1, 1].imshow(np.sum(sub_sat_pixels, axis=0)/float(sub_sat_pixels.shape[0]), vmin=0, vmax=1)
                axarr[1, 1].set_title('Saturated pixel map')
                c00 = plt.colorbar(p00, ax=axarr[0, 0])
                c00.set_label('Counts', rotation=270, labelpad=20)
                c01 = plt.colorbar(p01, ax=axarr[0, 1])
                c01.set_label('Counts', rotation=270, labelpad=20)
                c11 = plt.colorbar(p11, ax=axarr[1, 1])
                c11.set_label('Fraction of frames where pixel is saturated', rotation=270, labelpad=20)
                plt.savefig(self.cdir+cubes[i]+'_jitter.pdf', bbox_inches='tight')
                plt.show(block=block_plots)
                plt.close()
            
            # Save data cube
            fits_file = pyfits.open(self.cdir+cubes[i]+'_cleaned.fits')
            fits_file[0].header.add_comment('Subtracted '+cubes[furthest_cube[i]]+'_cleaned.fits')
            fits_file[0].data = sub_data
            fits_file[1].data = np.uint8(sub_bad_pixels)
            fits_file[2].data = np.uint8(sub_sat_pixels)
            fits_file.writeto(self.cdir+cubes[i]+'_jitter.fits', overwrite=True, output_verify='fix')
            
            logfile = open('log.txt', 'a')
            logfile.write('Data cube background subtracted and saved as '+cubes[i]+'_jitter.fits\n')
            logfile.close()
        
        pass
    
    def __bad_pixel_correction(self,
                               cubes):
        """
        Parameters
        ----------
        cubes : list
            List of cube paths
        """
        
        for i in range(len(cubes)):
            
            # Open data cube
            print('Processing data cube %.0f' % (i+1))
            fits_header = pyfits.getheader(self.cdir+cubes[i]+'_jitter.fits', 0)
            data = pyfits.getdata(self.cdir+cubes[i]+'_jitter.fits', 0)
            bad_pixels = pyfits.getdata(self.cdir+cubes[i]+'_jitter.fits', 1)
            sat_pixels = pyfits.getdata(self.cdir+cubes[i]+'_jitter.fits', 2)
            bgs = pyfits.getdata(self.cdir+cubes[i]+'_jitter.fits', 3)['BACKS']
            gain = fits_header['GAIN']
            readnoise = fits_header['RDNOISE']
            
            # Check if data cube has already been processed
            if (skip_processed_data):
                check = False
                processed_data = [f for f in os.listdir(self.cdir) if (fits_header['HIERARCH ESO OBS TARG NAME'] in f) and ('_bpcorrected.fits' in f)]
                for j in range(len(processed_data)):
                    if (fits_header['ORIGFILE'] == pyfits.getheader(self.cdir+processed_data[j])['ORIGFILE']):
                        check = True
                        break
                if (check):
                    print('Skipping data cube '+cubes[i]+', data cube has already been processed')
                    continue
            
            pmask = self.__pmask(fits_header=fits_header)
            fmask = self.__fmask(pmask=pmask)
            
            if (make_unimportant_plots):
                f, axarr = plt.subplots(1, 2, figsize=(12, 9))
                axarr[0].imshow(pmask)
                axarr[0].set_title('Pupil mask')
                axarr[1].imshow(fmask)
                axarr[1].set_title('Fourier mask')
                plt.show(block=block_plots)
                plt.close()
            
            if (make_plots):
                f, axarr = plt.subplots(2, 2, figsize=(12, 9))
                dummy = np.median(data, axis=0)
                p00 = axarr[0, 0].plot((dummy[sub_size/2, :]+dummy[:, sub_size/2])/2.)
                axarr[0, 0].set_title('PSF cross-section before correction')
                dummy = np.sum(sat_pixels, axis=0)/float(sat_pixels.shape[0])
                dummy[bad_pixels > 0.5] = 0
                p10 = axarr[1, 0].imshow(bad_pixels+dummy, vmin=0, vmax=1)
                axarr[1, 0].set_title('Bad pixel map before correction')
                c10 = plt.colorbar(p10, ax=axarr[1, 0])
                c10.set_label('Fraction of frames where pixel is bad', rotation=270, labelpad=20)
            
            # Identify and remove bad pixels
            bad_pixels_full = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
            for j in range(0, data.shape[0]):
                print('Removing bad pixels from frame %.0f' % (j+1))
                frame = data[j]
                frame_bad = bad_pixels.copy()
                frame_bad[sat_pixels[j] > 0.5] = 1
                
                dummy1 = frame.copy()
                dummy1[np.where(bad_pixels)] = 0.
                new_bad_pixels = frame_bad.copy()
                
                for loops in range(1, 15):
                    
                    # Correct known bad pixels
                    dummy2 = self.__fix_bad_pixels(dummy1, frame_bad, fmask)
                    
                    # Find extra bad pixels
                    extra_frame_ft = np.fft.rfft2(dummy2)*fmask
                    extra_frame = np.real(np.fft.irfft2(extra_frame_ft))
                    med_frame = nd.filters.median_filter(dummy2, size=5)
                    
                    # Using gain and readnoise from NACO user manual
                    total_noise = np.sqrt(np.maximum((bgs[j]+med_frame)/gain+readnoise**2, readnoise**2))
                    extra_frame /= total_noise
                    
                    # Subtract median filtered frame (ringing effect)
                    unsharp_masked = extra_frame-nd.filters.median_filter(extra_frame, size=3)
                    
                    # Find extra bad pixels based on variable threshold
                    # FIXME
                    extra_threshold = 7
                    current_threshold = np.max([0.25*np.max(np.abs(unsharp_masked[new_bad_pixels == 0])),\
                                                extra_threshold*np.median(np.abs(extra_frame))]) # Pixels above 1/4 of maximum or above 7 times median are bad
                    extra_bad_pixels = np.abs(unsharp_masked) > current_threshold
                    n_extra_bad_pixels = np.sum(extra_bad_pixels)
                    print(str(n_extra_bad_pixels)+' extra bad pixels identified (attempt '+str(loops)+'), threshold = '+str(current_threshold))
                    
                    # Add extra bad pixels to bad pixel map
                    frame_bad += extra_bad_pixels
                    frame_bad = frame_bad > 0.5
                    new_bad_pixels = extra_bad_pixels > 0.5
                    
                    # Break if no extra bad pixels were found
                    if (n_extra_bad_pixels == 0):
                        break
                print(str(np.sum(frame_bad))+' total bad pixels identified')
                
                # Correct all bad pixels
                data[j] = self.__fix_bad_pixels(data[j], frame_bad, fmask)
                bad_pixels_full[j] = frame_bad
                
            if (make_plots):
                dummy = np.median(data, axis=0)
                p01 = axarr[0, 1].plot((dummy[sub_size/2, :]+dummy[:, sub_size/2])/2.)
                axarr[0, 1].set_title('PSF cross-section after correction')
                axarr[0, 1].text(x=0.1*axarr[0, 1].get_xlim()[0], y=0.9*axarr[0, 1].get_ylim()[1], s='Noise std = %.1f' % np.std(dummy[pmask < 0.5]))
                p11 = axarr[1, 1].imshow(np.sum(bad_pixels_full, axis=0)/float(bad_pixels_full.shape[0]), vmin=0, vmax=1)
                axarr[1, 1].set_title('Bad pixel map after correction')
                c11 = plt.colorbar(p11, ax=axarr[1, 1])
                c11.set_label('Fraction of frames where pixel is bad', rotation=270, labelpad=20)
                plt.savefig(self.cdir+cubes[i]+'_bpcorrected.pdf', bbox_inches='tight')
                plt.show(block=block_plots)
                plt.close()
            
            # Save data cube
            fits_file = pyfits.open(self.cdir+cubes[i]+'_jitter.fits')
            fits_file[0].data = data
            fits_file[2].data = np.uint8(bad_pixels_full)
            fits_file[2].header['EXTNAME']  = 'FULL-BAD-PIXEL-MAP'
            del fits_file[2].header['COMMENT']
            fits_file[2].header.add_comment('Full bad pixel map')
            fits_file.writeto(self.cdir+cubes[i]+'_bpcorrected.fits', overwrite=True, output_verify='fix')
            
            logfile = open('log.txt', 'a')
            logfile.write('Data cube bad pixel corrected and saved as '+cubes[i]+'_bpcorrected.fits\n')
            logfile.close()
        
        pass

class dark(object):
    
    def __init__(self,
                 ddir,
                 rdir,
                 dark_paths):
        """
        Parameters
        ----------
        ddir : str
            Path of data directory
        rdir : str
            Path of reduction directory
        dark_paths : list
            List of dark paths
        """
        
        # Initialize directories
        self.ddir = ddir
        self.rdir = rdir
        
        # Define properties by which darks should be grouped
        props = {'NAXIS2': None, # Size in x-direction
                 'NAXIS1': None, # Size in y-direction
                 'EXPTIME': None} # Exposure time
        
        # Group darks by properties
        dark_paths_grouped, props_grouped, times_grouped = self.__group_by_props(dark_paths=dark_paths,
                                                                                 props=props)
        
        # Make master dark for each set of properties
        for i in range(len(dark_paths_grouped)):
            
            # Check if darks have already been processed
            if (self.__check(dark_paths_grouped=dark_paths_grouped[i],
                             props_grouped=props_grouped[i])):
                print('Skipping group %.0f, darks have already been processed' % (i+1))
                continue
            
            self.__make_master_dark(dark_paths_grouped=dark_paths_grouped[i],
                                    props_grouped=props_grouped[i],
                                    times_grouped=times_grouped[i])
        
        pass
    
    def __check(self,
                dark_paths_grouped,
                props_grouped):
        """
        Parameters
        ----------
        dark_paths_grouped : list
            List of grouped dark paths
        props_grouped : list
            List of grouped properties
        
        Returns
        -------
        check : bool
            True if darks have already been processed
        """
        
        # Output file name
        out_name = 'master_dark'
        for key in props_grouped.keys():
            out_name += '_'+str(props_grouped[key])
        
        # Check if darks have already been processed
        index = [f for f in os.listdir(self.rdir) if (out_name in f) and ('.fits' in f)]
        
        if (len(index) != 0):
            return True
        else:
            return False
    
    def __group_by_props(self,
                         dark_paths,
                         props):
        """
        Parameters
        ----------
        dark_paths : list
            List of dark paths
        props : dict
            Dictionary where keys are header keywords by which darks should be grouped
        
        Returns
        -------
        dark_paths_grouped : list
            List of lists of grouped dark paths
        props_grouped : list
            List of lists of grouped properties
        times_grouped : list
            List of lists of grouped observing times
        """
        
        # Initialize lists of lists of grouped dark paths, properties and observing times
        dark_paths_grouped = []
        props_grouped = []
        times_grouped = []
        
        # Open darks and read header
        for i in range(len(dark_paths)):
            try:
                fits_file = pyfits.open(self.ddir+dark_paths[i]+'.fits')
            except:
                try:
                    dark_paths[i] = dark_paths[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+dark_paths[i]+'.fits')
                except:
                    raise UserWarning('Could not find dark file '+dark_paths[i])
            fits_header = fits_file[0].header
            fits_file.close()
            
            # Extract properties from header
            props_temp = props.copy()
            for j in range(len(props)):
                props_temp[props.keys()[j]] = fits_header[props.keys()[j]]
            
            # Group darks by properties
            matched = False
            for j in range(len(props_grouped)):
                if (props_grouped[j] == props_temp):
                    dark_paths_grouped[j] += [dark_paths[i]]
                    times_grouped[j] += [fits_header['MJD-OBS']]
                    matched = True
            if (matched == False):
                dark_paths_grouped += [[dark_paths[i]]]
                props_grouped += [props_temp]
                times_grouped += [[fits_header['MJD-OBS']]]
        
#        # Verify output
#        for i in range(len(dark_paths_grouped)):
#            if (len(dark_paths_grouped[i]) != 3):
#                raise UserWarning('Darks should always be taken in sequences of three')
        
        # Return lists of lists of grouped dark paths, properties and observing times
        return dark_paths_grouped, props_grouped, times_grouped
    
    def __make_master_dark(self,
                           dark_paths_grouped,
                           props_grouped,
                           times_grouped):
        """
        Parameters
        ----------
        dark_paths_grouped : list
            List of grouped dark paths
        props_grouped : list
            List of grouped properties
        times_grouped : list
            List of grouped observing times
        """
        
        # Output file name
        out_name = 'master_dark'
        for key in props_grouped.keys():
            out_name += '_'+str(props_grouped[key])
        index = [f for f in os.listdir(self.rdir) if (out_name in f) and ('.fits' in f)]
        out_name += '_%03d' % len(index)
        
        # Open darks
        for i in range(len(dark_paths_grouped)):
            try:
                fits_file = pyfits.open(self.ddir+dark_paths_grouped[i]+'.fits')
            except:
                try:
                    dark_paths_grouped[i] = dark_paths_grouped[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+dark_paths_grouped[i]+'.fits')
                except:
                    raise UserWarning('Could not find dark file '+dark_paths_grouped[i])
            
            # Read header of first dark
            if (i == 0):
                fits_header = fits_file[0].header
            
            # Read data into array
            darks_temp = fits_file[0].data
            fits_file.close()
            if (len(darks_temp.shape) == 3):
                if (i == 0):
                    darks = darks_temp
                else:
                    darks = np.append(darks, darks_temp, axis=0)
            else:
                if (i == 0):
                    darks = np.zeros((1, darks_temp.shape[0], darks_temp.shape[1]))
                    darks[0] = darks_temp
                else:
                    dummy = np.zeros((1, darks_temp.shape[0], darks_temp.shape[1]))
                    dummy[0] = darks_temp
                    darks = np.append(darks, dummy, axis=0)
        print('Read darks of shape '+str(darks.shape))
        
        # Compute median and variance
        med_dark = np.median(darks, axis=0)
        if (linearize):
            print('Linearizing master dark')
            med_dark = linearize(frame=med_dark,
                                 pp=pp)
        print('Filtering with median filter of size 9')
        med_dark_filtered = nd.median_filter(med_dark, size=9)
        var_dark = np.var(darks, axis=0)
        
        dummy1 = np.abs(med_dark-med_dark_filtered)
        dummy2 = np.abs(var_dark-np.median(var_dark))
        
        # Find pixels with bad median or bad variance and saturated pixels
        med_mult = 10.
        med_diff = np.median(dummy1)
        print('Median difference: '+str(med_diff))
        bad_med = dummy1 > med_mult*med_diff
        print('Pixels with bad median: '+str(np.sum(bad_med)))
        var_mult = 75.        
        if (props_grouped['NAXIS1'] == 512):
            var_mult = 15.
        elif (props_grouped['NAXIS1'] == 256):
            var_mult = 8.
        med_var_diff = np.median(dummy2)
        print('Median variance difference: '+str(med_var_diff))
        bad_var = dummy2 > var_mult*med_var_diff
        print('Pixels with bad variance: '+str(np.sum(bad_var)))
        bad_pixels = np.logical_or(bad_med, bad_var, med_dark >= saturation_threshold)
        
        if (make_plots):
            f, axarr = plt.subplots(2, 2, figsize=(12, 9))
            p00 = axarr[0, 0].imshow(dummy1, vmin=0, vmax=med_mult*med_diff)
            axarr[0, 0].set_title('Absolute difference of median and filtered median')
            axarr[1, 0].hist(np.clip(dummy1.flatten(), 0, 35000), bins=50, range=(0, 35000))
            axarr[1, 0].axvline(med_mult*med_diff, color='red', label='bad pixel threshold')
            axarr[1, 0].set_yscale('log')
            axarr[1, 0].set_xlabel('Counts')
            axarr[1, 0].set_ylabel('Number of pixels')
            axarr[1, 0].legend()
            p01 = axarr[0, 1].imshow(dummy2, vmin=0, vmax=var_mult*med_var_diff)
            axarr[0, 1].set_title('Absolute difference of variance and median variance')
            axarr[1, 1].hist(np.clip(dummy2.flatten(), 0, 500), bins=50, range=(0, 500))
            axarr[1, 1].axvline(var_mult*med_var_diff, color='red', label='bad pixel threshold')
            axarr[1, 1].set_yscale('log')
            axarr[1, 1].set_xlabel('Counts')
            axarr[1, 1].set_ylabel('Number of pixels')
            axarr[1, 1].legend()
            c00 = plt.colorbar(p00, ax=axarr[0, 0])
            c00.set_label('Counts', rotation=270, labelpad=20)
            c01 = plt.colorbar(p01, ax=axarr[0, 1])
            c01.set_label('Counts', rotation=270, labelpad=20)
            plt.savefig(self.rdir+out_name+'.pdf', bbox_inches='tight')
            plt.show(block=block_plots)
            plt.close()
        
        # Compute readnoise
        for i in range(darks.shape[0]):
            darks[i] -= med_dark
        readnoise = np.std(darks, axis=0)
        readnoise = np.mean(readnoise)*np.sqrt(darks.shape[0]-1)/np.sqrt(darks.shape[0]-2)
        print('Readnoise: %.1f' % readnoise)
        
        # Save master dark
        pri_hdu = pyfits.PrimaryHDU(med_dark)
        pri_hdu.header = fits_header
        pri_hdu.header['EXTNAME'] = 'MASTER-DARK'
        pri_hdu.header['CAMERA'] = 'CONICA'
        pri_hdu.header['DATE-OBS'] = np.median(times_grouped)
        pri_hdu.header['PSCALE'] = float(fits_header['HIERARCH ESO INS PIXSCALE'])*1E3
        pri_hdu.header['CWAVE'] = float(fits_header['HIERARCH ESO INS CWLEN'])*1E-6
        pri_hdu.header['RDNOISE'] = float(readnoise)
        pri_hdu.header.add_comment('Master dark')
        for i in range(len(dark_paths_grouped)):
            pri_hdu.header.add_comment(dark_paths_grouped[i])
        
        sec_hdu = pyfits.ImageHDU(np.uint8(bad_pixels))
        sec_hdu.header['EXTNAME']  = 'BAD-PIXEL-MAP'
        sec_hdu.header.add_comment('Bad pixel map')
        
        out_file = pyfits.HDUList([pri_hdu, sec_hdu])
        out_file.writeto(self.rdir+out_name+'.fits', overwrite=True, output_verify='fix')
        
        pass

class flat(object):
    
    def __init__(self,
                 ddir,
                 rdir,
                 dark_paths,
                 flat_paths):
        """
        Parameters
        ----------
        ddir : str
            Path of data directory
        rdir : str
            Path of reduction directory
        dark_paths : list
            List of dark paths
        flat_paths : list
            List of flat paths
        """
        
        # Initialize directories
        self.ddir = ddir
        self.rdir = rdir
        
        # Define properties by which flats should be grouped
        props = {'NAXIS2': None, # Size in x-direction
                 'NAXIS1': None, # Size in y-direction
                 'EXPTIME': None, # Exposure time
                 'HIERARCH ESO INS CWLEN': None} # Filter wavelength
        
        # Group flats by properties
        flat_paths_grouped, props_grouped, times_grouped = self.__group_by_props(flat_paths=flat_paths,
                                                                                 props=props)
        
        # Make master flat for each set of properties
        for i in range(len(flat_paths_grouped)):
            
            # Check if flats have already been processed
            if (self.__check(flat_paths_grouped=flat_paths_grouped[i],
                             props_grouped=props_grouped[i])):
                print('Skipping group %.0f, flats have already been processed' % (i+1))
                continue
            
            self.__make_master_flat(flat_paths_grouped=flat_paths_grouped[i],
                                    props_grouped=props_grouped[i],
                                    times_grouped=times_grouped[i],
                                    dark_paths=dark_paths)
        
        pass
    
    def __check(self,
                flat_paths_grouped,
                props_grouped):
        """
        Parameters
        ----------
        flat_paths_grouped : list
            List of grouped flat paths
        props_grouped : list
            List of grouped properties
        
        Returns
        -------
        check : bool
            True if flats have already been processed
        """
        
        # Output file name
        out_name = 'master_flat'
        for key in props_grouped.keys():
            out_name += '_'+str(props_grouped[key])
        
        # Check if flats have already been processed
        index = [f for f in os.listdir(self.rdir) if (out_name in f) and ('.fits' in f)]
        
        if (len(index) != 0):
            return True
        else:
            return False
    
    def __find_master_dark(self,
                           props_grouped,
                           dark_paths):
        """
        Parameters
        ----------
        props_grouped : dict
            Dictionary of grouped properties
        dark_paths : list
            List of dark paths
        
        Returns
        -------
        dark_path : str
            Path of appropriate master dark
        """
        
        # Find path of appropriate master dark
        dark_path = [f for f in os.listdir(self.rdir) if ('master_dark' in f) and (str(props_grouped['NAXIS2']) in f) and (str(props_grouped['NAXIS1']) in f) and (str(props_grouped['EXPTIME']) in f) and ('.fits' in f)]
        if (len(dark_path) > 1):
            print('Identified more than 1 appropriate master darks, using first one')
        
        # Return path of appropriate master dark
        return dark_path[0]
    
    def __group_by_props(self,
                         flat_paths,
                         props):
        """
        Parameters
        ----------
        flat_paths : list
            List of flat paths
        props : dict
            Dictionary where keys are header keywords by which flat should be grouped
        
        Returns
        -------
        flat_paths_grouped : list
            List of lists of grouped flat paths
        props_grouped : list
            List of lists of grouped properties
        times_grouped : list
            List of lists of grouped observing times
        """
        
        # Initialize lists of lists of grouped flat paths, properties and observing times
        flat_paths_grouped = []
        props_grouped = []
        times_grouped = []
        
        # Open flats and read header
        for i in range(len(flat_paths)):
            try:
                fits_file = pyfits.open(self.ddir+flat_paths[i]+'.fits')
            except:
                try:
                    flat_paths[i] = flat_paths[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+flat_paths[i]+'.fits')
                except:
                    raise UserWarning('Could not find flat file '+flat_paths[i])
            fits_header = fits_file[0].header
            fits_file.close()
            
            # Verify input
            if (fits_header['HIERARCH ESO TPL NAME'] != 'LW Skyflats'):
                print(fits_header['HIERARCH ESO TPL NAME']+' flats will not be considered')
                continue
            
            # Extract properties from header
            props_temp = props.copy()
            for j in range(len(props)):
                props_temp[props.keys()[j]] = fits_header[props.keys()[j]]
            
            # Group flats by properties
            matched = False
            for j in range(len(props_grouped)):
                if (props_grouped[j] == props_temp):
                    flat_paths_grouped[j] += [flat_paths[i]]
                    times_grouped[j] += [fits_header['MJD-OBS']]
                    matched = True
            if (matched == False):
                flat_paths_grouped += [[flat_paths[i]]]
                props_grouped += [props_temp]
                times_grouped += [[fits_header['MJD-OBS']]]
        
        # Return lists of lists of grouped flat paths, properties and observing times
        return flat_paths_grouped, props_grouped, times_grouped
    
    def __make_master_flat(self,
                           flat_paths_grouped,
                           props_grouped,
                           times_grouped,
                           dark_paths):
        """
        Parameters
        ----------
        flat_paths_grouped : list
            List of grouped flat paths
        props_grouped : list
            List of grouped properties
        times_grouped : list
            List of grouped observing times
        dark_paths : list
            List of dark paths
        """
        
        # Output file name
        out_name = 'master_flat'
        for key in props_grouped.keys():
            out_name += '_'+str(props_grouped[key])
        index = [f for f in os.listdir(self.rdir) if (out_name in f) and ('.fits' in f)]
        out_name += '_%03d' % len(index)
        
        # Open flats
        for i in range(len(flat_paths_grouped)):
            try:
                fits_file = pyfits.open(self.ddir+flat_paths_grouped[i]+'.fits')
            except:
                try:
                    flat_paths_grouped[i] = flat_paths_grouped[i].replace(':', '_')
                    fits_file = pyfits.open(self.ddir+flat_paths_grouped[i]+'.fits')
                except:
                    raise UserWarning('Could not find flat file '+flat_paths_grouped[i])
            
            # Read header of first flat
            if (i == 0):
                fits_header = fits_file[0].header
            
            # Read data into array
            flats_temp = fits_file[0].data
            fits_file.close()
            if (len(flats_temp.shape) == 3):
                if (i == 0):
                    flats = flats_temp
                else:
                    flats = np.append(flats, flats_temp, axis=0)
            else:
                if (i == 0):
                    flats = np.zeros((1, flats_temp.shape[0], flats_temp.shape[1]))
                    flats[0] = flats_temp
                else:
                    dummy = np.zeros((1, flats_temp.shape[0], flats_temp.shape[1]))
                    dummy[0] = flats_temp
                    flats = np.append(flats, dummy, axis=0)
        print('Read flats of shape '+str(flats.shape))
        
        # Compute median and variance
        med_flat = np.median(flats, axis=0)
        if (linearize):
            print('Linearizing master flat')
            med_flat = linearize(frame=med_flat,
                                 pp=pp)
        print('Filtering with median filter of size 9')
        med_flat_filtered = nd.median_filter(med_flat, size=9)
        var_flat = np.var(flats, axis=0)
        
        dummy1 = np.abs(med_flat-med_flat_filtered)
        dummy2 = np.abs(var_flat-np.median(var_flat))
        
        # Find pixels with bad median or bad variance and saturated pixels
        med_mult = 7.5
        med_diff = np.median(dummy1)
        print('Median difference: '+str(med_diff))
        bad_med = dummy1 > med_mult*med_diff
        print('Pixels with bad median: '+str(np.sum(bad_med)))
        var_mult = 10.        
        med_var_diff = np.median(dummy2)
        print('Median variance difference: '+str(med_var_diff))
        bad_var = dummy2 > var_mult*med_var_diff
        print('Pixels with bad variance: '+str(np.sum(bad_var)))
        bad_pixels = np.logical_or(bad_med, bad_var, med_flat >= saturation_threshold)
        
        if (make_plots):
            f, axarr = plt.subplots(2, 2, figsize=(12, 9))
            p00 = axarr[0, 0].imshow(dummy1, vmin=0, vmax=med_mult*med_diff)
            axarr[0, 0].set_title('Absolute difference of median and filtered median')
            axarr[1, 0].hist(np.clip(dummy1.flatten(), 0, 25000), bins=50, range=(0, 25000))
            axarr[1, 0].axvline(med_mult*med_diff, color='red', label='bad pixel threshold')
            axarr[1, 0].set_yscale('log')
            axarr[1, 0].set_xlabel('Counts')
            axarr[1, 0].set_ylabel('Number of pixels')
            axarr[1, 0].legend()
            p01 = axarr[0, 1].imshow(dummy2, vmin=0, vmax=var_mult*med_var_diff)
            axarr[0, 1].set_title('Absolute difference of variance and median variance')
            axarr[1, 1].hist(np.clip(dummy2.flatten(), 0, 150000), bins=50, range=(0, 150000))
            axarr[1, 1].axvline(var_mult*med_var_diff, color='red', label='bad pixel threshold')
            axarr[1, 1].set_yscale('log')
            axarr[1, 1].set_xlabel('Counts')
            axarr[1, 1].set_ylabel('Number of pixels')
            axarr[1, 1].legend()
            c00 = plt.colorbar(p00, ax=axarr[0, 0])
            c00.set_label('Counts', rotation=270, labelpad=20)
            c01 = plt.colorbar(p01, ax=axarr[0, 1])
            c01.set_label('Counts', rotation=270, labelpad=20)
            plt.savefig(self.rdir+out_name+'.pdf', bbox_inches='tight')
            plt.show(block=block_plots)
            plt.close()
        
        # Find appropriate master dark
        dark_path = self.__find_master_dark(props_grouped=props_grouped,
                                            dark_paths=dark_paths)
        
        # Subtract master dark and normalize master flat
        med_flat_temp = med_flat-pyfits.getdata(self.rdir+dark_path, 0)
        bad_pixels_temp = np.logical_or(bad_pixels, pyfits.getdata(self.rdir+dark_path, 1) == 1)
        med_flat_temp /= np.median(med_flat_temp[bad_pixels_temp == 0])
        med_flat_temp[med_flat_temp == 0] = 1 # Remove stripes from broken quadrant (pixels where median dark == median flat)
        
        # Save master flat
        pri_hdu = pyfits.PrimaryHDU(med_flat_temp)
        pri_hdu.header = fits_header
        pri_hdu.header['EXTNAME'] = 'MASTER-FLAT'
        pri_hdu.header['CAMERA'] = 'CONICA'
        pri_hdu.header['DATE-OBS'] = np.median(times_grouped)
        pri_hdu.header['PSCALE'] = float(fits_header['HIERARCH ESO INS PIXSCALE'])*1E3
        pri_hdu.header['CWAVE'] = float(fits_header['HIERARCH ESO INS CWLEN'])*1E-6
        pri_hdu.header['RDNOISE'] = pyfits.getheader(self.rdir+dark_path)['RDNOISE']
        pri_hdu.header.add_comment('Master flat')
        for i in range(len(flat_paths_grouped)):
            pri_hdu.header.add_comment(flat_paths_grouped[i])
        pri_hdu.header.add_comment(dark_path)
        
        sec_hdu = pyfits.ImageHDU(np.uint8(bad_pixels_temp))
        sec_hdu.header['EXTNAME']  = 'BAD-PIXEL-MAP'
        sec_hdu.header.add_comment('Bad pixel map')
        
        out_file = pyfits.HDUList([pri_hdu, sec_hdu])
        out_file.writeto(self.rdir+out_name+'.fits', overwrite=True, output_verify='fix')
        
        print('Looking for master darks of different size')
        master_darks = [f for f in os.listdir(self.rdir) if ('master_dark' in f) and ('.fits' in f)]
        master_darks_temp = []
        for i in range(len(master_darks)):
            
            # Find master darks of different size
            if ((str(props_grouped['NAXIS2']) not in master_darks[i]) and (str(props_grouped['NAXIS1']) not in master_darks[i])):
                dark_header = pyfits.getheader(self.rdir+master_darks[i])
                similar_exptime = False
                for j in range(len(master_darks)):
                    
                    # Find master darks of same exposure time
                    if ((j != i) and (str(dark_header['NAXIS2']) in master_darks[j]) and (str(dark_header['NAXIS1']) in master_darks[j]) and (str(props_grouped['EXPTIME']) in master_darks[j])):
                        similar_exptime = True
                if (similar_exptime == False):
                    master_darks_temp += [master_darks[i]]
        master_darks = master_darks_temp
        del master_darks_temp
        
        for i in range(len(master_darks)):
            dark_header = pyfits.getheader(self.rdir+master_darks[i])
            
            x_size = int(dark_header['NAXIS2'])
            y_size = int(dark_header['NAXIS1'])
            print('Identified master dark of size %.0f x %.0f' % (x_size, y_size))
            
            # Crop master flat to size of master dark
            if (dark_header['NAXIS2'] < props_grouped['NAXIS2'] and dark_header['NAXIS1'] < props_grouped['NAXIS1']):
                x_half = int(props_grouped['NAXIS2'])/2
                y_half = int(props_grouped['NAXIS1'])/2
                med_flat_temp = med_flat[x_half-x_size/2:x_half+x_size/2, y_half-y_size/2:y_half+y_size/2]
                bad_pixels_temp = bad_pixels[x_half-x_size/2:x_half+x_size/2, y_half-y_size/2:y_half+y_size/2]
            else:
                print('Skipping master dark because it is too large')
                continue
            
            # Output file name
            out_name = 'master_flat'
            props_grouped_temp = props_grouped.copy()
            props_grouped_temp['NAXIS2'] = x_size
            props_grouped_temp['NAXIS1'] = y_size
            for key in props_grouped_temp.keys():
                out_name += '_'+str(props_grouped_temp[key])
            index = [f for f in os.listdir(self.rdir) if (out_name in f) and ('.fits' in f)]
            out_name += '_%03d' % len(index)
            
            # Subtract master dark and normalize master flat
            ratio = float(fits_header['EXPTIME'])/float(dark_header['EXPTIME'])
            med_flat_temp = med_flat_temp-pyfits.getdata(self.rdir+master_darks[i], 0)*ratio
            bad_pixels_temp = np.logical_or(bad_pixels_temp, pyfits.getdata(self.rdir+master_darks[i], 1) == 1)
            med_flat_temp /= np.median(med_flat_temp[bad_pixels_temp == 0])
            med_flat_temp[med_flat_temp == 0] = 1 # Remove stripes from broken quadrant (pixels where median dark == median flat)
            
            # Save master flat
            pri_hdu = pyfits.PrimaryHDU(med_flat_temp)
            pri_hdu.header = fits_header
            del pri_hdu.header['COMMENT']
            pri_hdu.header['EXTNAME'] = 'MASTER-FLAT'
            pri_hdu.header['CAMERA'] = 'CONICA'
            pri_hdu.header['DATE-OBS'] = np.median(times_grouped)
            pri_hdu.header['PSCALE'] = float(fits_header['HIERARCH ESO INS PIXSCALE'])*1E3
            pri_hdu.header['CWAVE'] = float(fits_header['HIERARCH ESO INS CWLEN'])*1E-6
            pri_hdu.header['RDNOISE'] = pyfits.getheader(self.rdir+master_darks[i])['RDNOISE']
            pri_hdu.header.add_comment('Master flat')
            for j in range(len(flat_paths_grouped)):
                pri_hdu.header.add_comment(flat_paths_grouped[j])
            pri_hdu.header.add_comment(master_darks[i])
            
            sec_hdu = pyfits.ImageHDU(np.uint8(bad_pixels_temp))
            sec_hdu.header['EXTNAME']  = 'BAD-PIXEL-MAP'
            sec_hdu.header.add_comment('Bad pixel map')
            
            out_file = pyfits.HDUList([pri_hdu, sec_hdu])
            out_file.writeto(self.rdir+out_name+'.fits', overwrite=True, output_verify='fix')
        
        pass
    