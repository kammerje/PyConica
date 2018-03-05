#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 8 2018

@author: Jens Kammerer
"""


import core_new as core
import os
import time


ddir = '/priv/mulga2/kjens/NACO/girard/data_with_raw_calibs/'
rdir = '/priv/mulga2/kjens/NACO/girard/test/'
cdir = '/priv/mulga2/kjens/NACO/girard/test/'
index = 0

#test_dir = '/home/kjens/Downloads/'
#cube = core.cube(ddir=ddir,
#                 rdir=test_dir,
#                 cdir=test_dir,
#                 xml_path='NACO.2016-06-15T05:46:13.750.xml')

xml_files = [f for f in os.listdir(ddir) if f.endswith('.xml')]
print('IDENTIFIED '+str(len(xml_files))+' XML FILES')
#time.sleep(10)

#start_file = 'NACO.2016-07-22T08:43:59.213.xml'
#for index in range(0, len(xml_files)):
#    if (start_file == xml_files[index]):
#        break
#print('Starting with xml file '+xml_files[index])

times = [time.time()]
for i in range(index, len(xml_files)):
    rdir_now = rdir+'ds%03d' % i+'/'
    if (os.path.exists(rdir_now) == False):
        os.makedirs(rdir_now)
    
    file = open('log.txt', 'a')
    file.write('Processing '+xml_files[i]+', file %03d' % i+' of %03d' % len(xml_files)+'\n')
    file.close()
    
    cube = core.cube(ddir=ddir,
                     rdir=rdir_now,
                     cdir=cdir,
                     xml_path=xml_files[i])
    times += [time.time()]
    runtime = times[-1]-times[-2]
    
    file = open('log.txt', 'a')
    file.write('Finished in %06d' % runtime+' seconds\n')
    file.close()
