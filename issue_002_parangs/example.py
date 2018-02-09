#==============================================================================
# Calculates parallactic angles
# (is the equation in lines 45 - 47 correct?)
#==============================================================================

import astropy.io.fits as pyfits
import numpy as np

# Data directory
data_path = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-07-11T06:49:00.538.fits'

"""
This is the code from my pipeline. It extracts information from the header and
calculates the parallactic angle for each frame by linear interpolation between
the start and end values. Is the equation in lines 45 - 47 correct?
"""

# Get data cube
fits_file = pyfits.open(data_path)
fits_header = fits_file[0].header
arrays = fits_file[0].data[:-1, :, :] # Last frame is mean, throw it away
fits_file.close()
print('Loaded data cube '+data_path+' of shape '+str(arrays.shape))

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

# Get parallactic angle
parangs = np.zeros(arrays.shape[0])
for i in range(arrays.shape[0]):
    parangs[i] = alt+instrument_offset+\
                 (rot_start+(rot_end-rot_start)/float(arrays.shape[0])*float(i))-\
                 (180.-(parang_start+(parang_end-parang_start)/float(arrays.shape[0])*float(i))) # FIX

print(parangs) # In degrees I assume
