"""
PyConica, a Python package to reduce NACO data. Information about NACO can be
found at https://www.eso.org/sci/facilities/paranal/instruments/naco.html. This
library is maintained on GitHub at https://github.com/kammerje/PyConica.

Author: Jens Kammerer
Version: 1.0.3
Last edited: 20.07.18
"""


# PREAMBLE
#==============================================================================
import sys
sys.path.append('/home/kjens/Python/Packages/opticstools/opticstools/opticstools')

import astropy.io.fits as pyfits
import heapq as hq
import matplotlib.pyplot as plt
import numpy as np
import opticstools as ot
import os
import time


# PARAMETERS
#==============================================================================
make_plots = True
make_unimportant_plots = False
block_plots = False

cdir = '/priv/mulga2/kjens/NACO/girard_2016/cubes/'
odir = '/priv/mulga2/kjens/NACO/girard_2016/lucky/'

sub_size = 96


# MAIN
#==============================================================================
print('--> Scanning cubes directory for bad pixel corrected fits files')
fits_paths = [f for f in os.listdir(cdir) if f.endswith('bpcorrected.fits')]
print('Identified '+str(len(fits_paths))+' bad pixel corrected fits files')
time.sleep(3)

def __pmask(fits_header):
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
    
    # Set telescope dimensions (noise will be computed outside the pupil mask, therefore use extra big mask to avoid residuals from the PSF)
    mirror_size = 12 # meters
    
    # Extract relevant information from header
    pscale = float(fits_header['PSCALE'])
    cwave = float(fits_header['CWAVE'])
    
    # Set scale ratio for Fourier transformation
    ratio = cwave/(pscale/1000./60./60.*np.pi/180.*sub_size)
    
    # Make pupil mask
    pmask = ot.circle(sub_size, mirror_size/ratio)
    
    # Return pupil mask
    return pmask

for i in range(len(fits_paths)):
    
    # Open data cube and read header
    try:
        fits_file = pyfits.open(cdir+fits_paths[i])
    except:
        raise UserWarning('Could not find fits file '+fits_paths[i])
    fits_header = fits_file[0].header
    
    #
    if (i == 0):
        pmask = __pmask(fits_header=fits_header)
        if (make_unimportant_plots):
            plt.figure(figsize=(12, 9))
            plt.imshow(pmask)
            plt.title('Pupil mask')
            plt.show(block=block_plots)
            plt.close()
    
    # Read data into array
    data = fits_file[0].data
    print('Read data of shape '+str(data.shape))
    
    # Extract peak flux and noise from each frame
    peak = []
    noise = []
    for j in range(data.shape[0]):
        peak += [np.max(data[j])]
        noise += [np.std(data[j][pmask < 0.5])]
    SNR = np.true_divide(peak, noise)
    
    # Compute peak flux, noise and SNR thresholds
    peak_cut = 0.75*np.median(hq.nlargest(int(0.1*data.shape[0]), peak))
    noise_cut = 1.1*np.median(hq.nsmallest(int(0.1*data.shape[0]), noise))
    SNR_cut = 0.75*np.median(hq.nlargest(int(0.1*data.shape[0]), SNR))
    
    if (make_plots):
        f, axarr = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
        axarr[0].plot(peak)
        axarr[0].axhline(peak_cut, color='red', label='Rejection threshold')
        axarr[0].set_ylabel('Peak flux [ADU]')
        axarr[0].legend(loc='upper right')
        axarr[1].plot(noise)
        axarr[1].axhline(noise_cut, color='red', label='Rejection threshold')
        axarr[1].set_ylabel('Noise [standard deviation of ADU]')
        axarr[1].legend(loc='upper right')
        axarr[2].plot(SNR)
        axarr[2].axhline(SNR_cut, color='red', label='Rejection threshold')
        axarr[2].set_ylabel('SNR')
        axarr[2].set_xlabel('Frame number')
        axarr[2].legend(loc='upper right')
        plt.suptitle('Lucky imaging')
        plt.savefig(odir+fits_paths[i][:-17]+'_lucky.pdf', bbox_inches='tight')
        plt.show(block=block_plots)
        plt.close()
    
    if (make_unimportant_plots):
        plt.figure()
        plt.imshow(np.median(data, axis=0)+50*pmask, vmin=-50, vmax=50)
        plt.show(block=block_plots)
        plt.close()
    
    # Only keep frames which have good peak count and good SNR
    mask1 = peak > peak_cut
#    mask2 = SNR > SNR_cut
#    mask = mask1 & mask2
    mask = mask1
    
    # Save data cube
    fits_file[0].data = fits_file[0].data[mask]
    fits_file[2].data = fits_file[2].data[mask]
    fits_file[3].data = fits_file[3].data[mask]
    fits_file[0].header.add_comment('LuckyImaging: Rejected frames '+str(list(np.where(mask == False)[0])))
    fits_file.writeto(odir+fits_paths[i][:-17]+'_lucky.fits', overwrite=True, output_verify='fix')
