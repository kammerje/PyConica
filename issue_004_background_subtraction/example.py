#==============================================================================
# Makes background subtraction
# (is the very simple approach in lines 46 - 48 sufficient?)
#==============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

# Data directory
dst = '/priv/mulga2/kjens/test/'

"""
The options make_plots and block_plots will show and block plots of the data
cube before and after the background subtraction.
"""

make_plots = True
block_plots = True

"""
This is the code from my pipeline. It loads the dark and flat subtracted data
cube and goes on with the background subtraction. Is the very simple approach
in lines 46 - 48 sufficient?
"""

# Get dark and flat subtracted data cube
fits_file = pyfits.open(dst+'cube_002.fits')
fits_header = fits_file[0].header
arrays = fits_file[0].data
bad_pixels = fits_file[1].data
fits_file.close()
print('Loaded data cube '+dst+'cube_002.fits'+' of shape '+str(arrays.shape))

if (make_plots == True):
    plt.figure()
    plt.imshow(np.median(arrays, axis=0))
    plt.title('Median of data cube before bg subtraction')
    plt.show(block=block_plots)

# Subtract background
backgrounds = np.zeros(arrays.shape[0])
for i in range(arrays.shape[0]):
    if ((i+1)%10 == 0):
        print('Cleaning frame '+str(i+1))
    frame = arrays[i, :, :]
    
    # Subtract background
    backgrounds[i] = np.median(frame[bad_pixels == 0])
    frame -= backgrounds[i]
    
    arrays[i, :, :] = frame

if (make_plots == True):
    plt.figure()
    plt.imshow(np.median(arrays, axis=0))
    plt.title('Median of data cube after bg subtraction')
    plt.show(block=block_plots)

# Save data cube
fits_header['NAXIS2'] = arrays.shape[1]
fits_header['NAXIS1'] = arrays.shape[2]
fits_header['NAXIS3'] = arrays.shape[0]
out_name = 'cube_002_bg_subtracted'
out_file = pyfits.HDUList()
out_file.append(pyfits.ImageHDU(arrays, fits_header))
out_file.append(pyfits.ImageHDU(np.uint8(bad_pixels)))
out_file.writeto(dst+out_name+'.fits',\
                 output_verify='ignore',\
                 overwrite=True)
