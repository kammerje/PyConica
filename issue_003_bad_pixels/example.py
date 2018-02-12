#==============================================================================
# Identifies and removes the bad pixels
# (last plot shows that extra bad pixels are only identified close to the
# target. Also pay attention to the strange dot on e.g. frame 18 of the cleaned
# data cube)
#==============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd

# Data directory
dst = '/priv/mulga2/kjens/test/'

# Choose between one of the data sets. For data set 000, the final data has
# strange dots, for data set 001 it looks good
ds = '000'
#ds = '001'

"""
The options make_plots and block_plots will show and block plots of the Fourier
mask, the data cube, the master flat and the the bad pixel map.
"""

make_plots = True
block_plots = True

# Some stuff we need
sub_size = 128
backgrounds = np.load(dst+'backgrounds_'+ds+'.npy')
max_x = np.load(dst+'max_x_'+ds+'.npy')
max_y = np.load(dst+'max_y_'+ds+'.npy')
fmask = np.load(dst+'fmask_'+ds+'.npy')
gain = 9.8
read_noise = 4.4

if (make_plots == True):
    plt.figure()
    plt.imshow(fmask)
    plt.title('Fourier mask')
    plt.show(block_plots)

"""
This is the code from my pipeline. It starts with two function definitions, of
which I added the first one to the pynrm code myself. It identifies the stripes
on the broken quadrant of Conica.
"""

def _remove_stripes(bad_pixels):
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

def _fix_bad_pixels(frame,
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

"""
In the following, the data is loaded and the bad pixels are identified and
removed from the data. The last plot shows the sum of the 3-dimensional bad
pixel cube. The highest values are bad pixels propagated from the darks and the
flats which are on each frame, the rest are extra bad pixels identified during
the iterative process. The problem is, that such extra bad pixels are only
identified close to the target. Also pay attention to the strange dot on e.g.
frame 18 of the cleaned data cube.
"""

# Get dark, flat and background subtracted data cube
fits_file = pyfits.open(dst+'cube_'+ds+'.fits')
fits_header = fits_file[0].header
arrays = fits_file[0].data
bad_pixels = fits_file[1].data
fits_file.close()
print('Loaded data cube '+dst+'cube_'+ds+'.fits'+' of shape '+str(arrays.shape))

if (make_plots == True):
    plt.figure()
    plt.imshow(np.median(arrays, axis=0))
    plt.title('Median of data cube')
    plt.show(block_plots)

# Get master flat
fits_file = pyfits.open(dst+'master_flat_'+ds+'.fits')
fits_header = fits_file[0].header
flat = fits_file[0].data
fits_file.close()
print('Loaded master flat '+dst+'master_flat_'+ds+'.fits'+' of shape '+str(flat.shape))

if (make_plots == True):
    plt.figure()
    plt.imshow(flat)
    plt.title('Master flat')
    plt.show(block_plots)

# Cut out sub-arrays
print('Cutting out sub-arrays')
sub_arrays = np.zeros((arrays.shape[0], sub_size, sub_size))
sub_arrays_bad_pixels = np.zeros((arrays.shape[0], sub_size, sub_size))
sub_flat = np.roll(np.roll(flat,\
           int(sub_size/2.)-max_y, axis=0),\
           int(sub_size/2.)-max_x, axis=1)[0:sub_size, 0:sub_size]

if (make_plots == True):
    plt.figure()
    plt.imshow(bad_pixels)
    plt.title('Bad pixel map before stripes removal')
    plt.show(block_plots)

# Remove stripes from bad pixel map
bad_pixels, stripes_map = _remove_stripes(bad_pixels)
sub_stripes_map = np.roll(np.roll(stripes_map,\
                                  int(sub_size/2.)-max_y, axis=0),\
                                  int(sub_size/2.)-max_x, axis=1)[0:sub_size, 0:sub_size]

if (make_plots == True):
    plt.figure()
    plt.imshow(bad_pixels)
    plt.title('Bad pixel map after stripes removal')
    plt.show(block_plots)

# Remove bad pixels
for i in range(0, arrays.shape[0]):
    print('Removing bad pixels from frame '+str(i+1))
    frame = arrays[i, :, :]*flat # Undo flat-fielding
    sub_arrays[i, :, :] = np.roll(np.roll(frame,\
                          int(sub_size/2.)-max_y, axis=0),\
                          int(sub_size/2.)-max_x, axis=1)[0:sub_size, 0:sub_size]
    sub_arrays[i, :, :] = np.true_divide(sub_arrays[i, :, :], sub_flat) # Redo flat-fielding
    sub_bad_pixels = np.roll(np.roll(bad_pixels,\
                     int(sub_size/2.)-max_y, axis=0),\
                     int(sub_size/2.)-max_x, axis=1)[0:sub_size, 0:sub_size]
        
    # Find bad pixels frame
    new_bad_pixels = sub_bad_pixels.copy()
    dummy1 = sub_arrays[i, :, :].copy()
    dummy1[np.where(sub_bad_pixels)] = 0.
    
    for loops in range(1, 15):
        
        # Correct known bad pixels
        dummy2 = _fix_bad_pixels(dummy1, sub_bad_pixels, fmask)
        
        # Find extra bad pixels
        extra_frame_ft = np.fft.rfft2(dummy2)*fmask
        extra_frame = np.real(np.fft.irfft2(extra_frame_ft))
        frame_median = nd.filters.median_filter(dummy2, size=5)
        
        # Using gain and read noise from NACO user manual
        total_noise = np.sqrt(np.maximum((backgrounds[i]+frame_median)/gain+read_noise**2,\
                                         read_noise**2))
        extra_frame /= total_noise
        
        # Subtract median filtered frame (ringing effect)
        unsharp_masked = extra_frame-nd.filters.median_filter(extra_frame, size=3)
        
        # Find extra bad pixels based on variable threshold
        extra_threshold = 7 # USER
        current_threshold = np.max([0.25*np.max(np.abs(unsharp_masked[new_bad_pixels < 0.5])),\
                                    extra_threshold*np.median(np.abs(extra_frame))]) # Pixels above 1/4 of the maximum or above 7 times the median are bad
        extra_bad_pixels = np.abs(unsharp_masked) > current_threshold
        """
        The next line prevents stripes on the broken quadrant of Conica to be
        identified as bad pixels.
        """
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
    sub_arrays[i, :, :] = _fix_bad_pixels(sub_arrays[i, :, :], sub_bad_pixels, fmask)
    sub_arrays_bad_pixels[i, :, :] = sub_bad_pixels

if (make_plots == True):
    plt.figure()
    plt.imshow(np.sum(sub_arrays_bad_pixels, axis=0))
    plt.title('Sum of 3-dimensional bad pixel cube')
    plt.show(block=block_plots)

# Save data cube
fits_header['NAXIS2'] = sub_arrays.shape[1]
fits_header['NAXIS1'] = sub_arrays.shape[2]
fits_header['NAXIS3'] = sub_arrays.shape[0]
out_name = 'cube_'+ds+'_cleaned'
out_file = pyfits.HDUList()
out_file.append(pyfits.ImageHDU(sub_arrays, fits_header))
out_file.append(pyfits.ImageHDU(np.uint8(sub_arrays_bad_pixels)))
out_file.writeto(dst+out_name+'.fits',\
                 output_verify='ignore',\
                 overwrite=True)
