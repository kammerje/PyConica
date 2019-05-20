"""
PyConica, a Python package to reduce NACO data. Information about NACO can be
found at https://www.eso.org/sci/facilities/paranal/instruments/naco.html. This
library is maintained on GitHub at https://github.com/kammerje/PyConica.

Author: Jens Kammerer
Version: 1.2.0
Last edited: 20.11.18
"""


# PREAMBLE
#==============================================================================
import PyConica as PyCo

import os
import time


# DIRECTORIES
#==============================================================================
ddir = '/priv/mulga2/kjens/NACO/girard_2016/data_with_raw_calibs/'
rdir = '/priv/mulga2/kjens/NACO/girard_2016/redux/'
cdir = '/priv/mulga2/kjens/NACO/girard_2016/cubes/'


# MAIN
#==============================================================================
print('--> Scanning data directory for xml files')
xml_paths = [f for f in os.listdir(ddir) if f.endswith('.xml')]
print('Identified '+str(len(xml_paths))+' xml files')
time.sleep(3)

times = [time.time()]
for i in range(len(xml_paths)):
#for i in [20]:
    
    logfile = open('log.txt', 'a')
    logfile.write('--> Processing '+xml_paths[i]+', file %03d' % (i+1)+' of %03d' % (len(xml_paths))+'\n')
    logfile.close()
    
    rdir_temp = rdir+'ds%03d' % i+'/'
    if (os.path.exists(rdir_temp) == False):
        os.makedirs(rdir_temp)
    
    cube = PyCo.cube(ddir=ddir,
                     rdir=rdir_temp,
                     cdir=cdir)
    cube.process_ob(xml_path=xml_paths[i])
    
    times += [time.time()]
    runtime = times[-1]-times[-2]
    
    logfile = open('log.txt', 'a')
    logfile.write('Finished in %06d' % runtime+' seconds\n')
    logfile.close()
