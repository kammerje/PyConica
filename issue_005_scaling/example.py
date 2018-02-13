#==============================================================================
# First frame of cleaned data cubes shows different scaling
# (What is the reason for that and what can we do about it?)
#==============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

# Data directory
dst = '/priv/mulga2/kjens/test/'

"""
Looking at the cleaned data cubes (with e.g. DS9), the first frame always shows
a different scaling (use scale: log and color: cool).
"""

# Look at this cleaned data cube to see different scaling of first frame
fits_file = pyfits.open(dst+'cube_000_cleaned.fits')
fits_file.close()

"""
Choose one of the following two sets of raw data cubes. The first one
corresponds to the cleaned data cube above, but looking at the mean of each
frame of the raw data cube does not really indicate that the issue with the
scaling is an issue with the raw data. However, if looking at the second set of
raw data cubes, it seems to be obvious that something must be different with
the first frame, probably leading to the observed different scaling.
"""

# Raw data cubes
raw1 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-07-11T06:49:00.538.fits'
raw2 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-07-11T06:49:34.328.fits'
raw3 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-07-11T06:50:06.319.fits'
raw4 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-07-11T06:50:38.309.fits'
#raw1 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-09-19T01:16:18.506.fits'
#raw2 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-09-19T01:16:57.695.fits'
#raw3 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-09-19T01:17:36.150.fits'
#raw4 = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/NACO.2016-09-19T01:18:15.537.fits'

# Load multiple raw data cubes
fits_file = pyfits.open(raw1)
data1 = fits_file[0].data
fits_file.close()
fits_file = pyfits.open(raw2)
data2 = fits_file[0].data
fits_file.close()
fits_file = pyfits.open(raw3)
data3 = fits_file[0].data
fits_file.close()
fits_file = pyfits.open(raw4)
data4 = fits_file[0].data
fits_file.close()

# Clearly, last frame is mean
plt.figure()
plt.imshow(np.mean(data1[:-1, :, :], axis=0)-data1[-1, :, :])
plt.title('Proof that last frame is mean')
plt.show(block=True)

# Plot mean of each frame of multiple raw data cubes
data1 = data1[:-1, :, :]
data2 = data2[:-1, :, :]
data3 = data3[:-1, :, :]
data4 = data4[:-1, :, :]
plt.figure()
plt.plot(np.mean(data1, axis=(1, 2)))
plt.plot(np.mean(data2, axis=(1, 2)))
plt.plot(np.mean(data3, axis=(1, 2)))
plt.plot(np.mean(data4, axis=(1, 2)))
plt.title('First frame has higher mean than next few frames')
plt.show(block=True)
