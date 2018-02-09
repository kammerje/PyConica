#==============================================================================
# Copy relevant fits files from server
# (must not be executed)
#==============================================================================

#from shutil import copyfile
#from xml.dom import minidom
#
## Path of ESO xml file
#xml_path = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-07-11T06:49:00.538.xml'
#
## Open ESO xml file
#xml_file = minidom.parse(xml_path)
#
## Get all items of type file
#xml_items = xml_file.getElementsByTagName('file')
#
## Identify data, dark and flat items
#dark_paths = []
#flat_paths = []
#data_paths = []
#for item in xml_items:
#    if (item.attributes['category'].value == 'IM_JITTER_OBJ'):
#        data_paths += [item.attributes['name'].value]
#    elif (item.attributes['category'].value == 'CAL_DARK'):
#        dark_paths += [item.attributes['name'].value]
#    elif (item.attributes['category'].value == 'CAL_FLAT_TW'):
#        flat_paths += [item.attributes['name'].value]
#    else:
#        print('Could not match '+item.attributes['category'].value)
#
#print(str(len(dark_paths))+' dark files identified')
#print(str(len(flat_paths))+' flat files identified')
#print(str(len(data_paths))+' data files identified')
#
## Copy relevant fits files from src to dst
#src = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/'
#dst = '/priv/mulga2/kjens/test'
#for i in range(0, len(dark_paths)):
#    copyfile(src+dark_paths[i]+'.fits', dst+'dark_%03d' % i+'.fits')
#for i in range(0, len(flat_paths)):
#    copyfile(src+flat_paths[i]+'.fits', dst+'flat_%03d' % i+'.fits')


#==============================================================================
# Make master darks
# (illustrates problem with 1024x1024 darks)
#==============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

# Data directory
dst = '/priv/mulga2/kjens/test'

"""
Comment out one of the two dark_lists to see how the bad pixel identification
works for the remaining one. The options make_plots and block_plots will show
and block plots of the final median dark and its bad pixel map.
"""

# 520x512 darks, bad pixel identification works reasonably well
#dark_list = ['dark_000.fits', 'dark_001.fits', 'dark_002.fits']

# 1024x1024 darks, bad pixel identification doesn't work
dark_list = ['dark_003.fits', 'dark_004.fits', 'dark_005.fits']

make_plots = True
block_plots = True

"""
This is the code from my pipeline. It loads all darks from the dark_list into
the 3-dimensional array darks and calculates the final median dark and its bad
pixel map.
"""

# Get darks
for i in range(0, len(dark_list)):
    fits_file = pyfits.open(dst+dark_list[i])
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
max_dark = np.max(darks, axis=0)
var_dark = np.zeros((darks.shape[1], darks.shape[2]))
for i in range(0, darks.shape[0]):
    var_dark += (darks[i, :, :]-med_dark)**2.
var_dark -= (max_dark-med_dark)**2. # Throw away maximum
var_dark /= (darks.shape[0]-2.)

# Find pixels with bad median or bad variance
med_diff = np.maximum(np.median(np.abs(med_dark-np.median(med_dark))), 1.)
print('Median difference: '+str(med_diff))
med_var_diff = np.median(np.abs(var_dark-np.median(var_dark)))
print('Median variance difference: '+str(med_var_diff))
bad_med = np.abs(med_dark-np.median(med_dark)) > 15.*med_diff # USER
bad_var = var_dark > np.median(var_dark)+10.*med_var_diff     # USER
print('Pixels with bad median: '+str(np.sum(bad_med)))
print('Pixels with bad variance: '+str(np.sum(bad_var)))
bad_pixels = np.logical_or(bad_med, bad_var)
med_dark[bad_pixels] = 0.

if (make_plots == True):
    plt.figure()
    plt.imshow(med_dark)
    plt.show(block=block_plots)

if (make_plots == True):
    plt.figure()
    plt.imshow(bad_pixels)
    plt.show(block=block_plots)


#==============================================================================
# Make master flats
# (illustrates problem with 1024x1024 flats)
#==============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

# Data directory
dst = '/priv/mulga2/kjens/test'

"""
The options make_plots and block_plots will show and block plots of the final
median flat and its bad pixel map.
"""

# 1024x1024 flats, there are many bad pixels identified in one corner
flat_list = ['flat_000.fits', 'flat_001.fits', 'flat_002.fits',\
             'flat_003.fits', 'flat_004.fits', 'flat_005.fits',\
             'flat_006.fits', 'flat_007.fits', 'flat_008.fits',\
             'flat_009.fits', 'flat_010.fits', 'flat_011.fits',\
             'flat_012.fits', 'flat_013.fits', 'flat_014.fits']

make_plots = True
block_plots = True

"""
This is the code from my pipeline. It loads all flats from the flat_list into
the 3-dimensional array flats and calculates the final median flat and its bad
pixel map.
"""

# Get flats
for i in range(0, len(flat_list)):
    fits_file = pyfits.open(dst+flat_list[i])
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
max_flat = np.max(flats, axis=0)
var_flat = np.zeros((flats.shape[1], flats.shape[2]))
for i in range(0, flats.shape[0]):
    var_flat += (flats[i, :, :]-med_flat)**2.
var_flat -= (max_flat-med_flat)**2. # Throw away maximum
var_flat /= (flats.shape[0]-2.)

# Find pixels with bad median or bad variance
med_diff = np.maximum(np.median(np.abs(med_flat-np.median(med_flat))), 1.)
print('Median difference: '+str(med_diff))
med_var_diff = np.median(np.abs(var_flat-np.median(var_flat)))
print('Median variance difference: '+str(med_var_diff))
bad_med = np.abs(med_flat-np.median(med_flat)) > 15.*med_diff # USER
bad_var = var_flat > np.median(var_flat)+10.*med_var_diff     # USER
print('Pixels with bad median: '+str(np.sum(bad_med)))
print('Pixels with bad variance: '+str(np.sum(bad_var)))
bad_pixels = np.logical_or(bad_med, bad_var)
med_flat[bad_pixels] = 0.

if (make_plots == True):
    plt.figure()
    plt.imshow(med_flat)
    plt.show(block=block_plots)

if (make_plots == True):
    plt.figure()
    plt.imshow(bad_pixels)
    plt.show(block=block_plots)
