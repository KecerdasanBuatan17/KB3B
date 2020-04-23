# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:50:18 2020

@author: Liyana
"""

import os
import eyed3

root_folder = 'D:/mata kuliah poltekppos/Data Tingkat 3/Semester 6/Kecerdasan Buatan/uts/lagu'

files = os.listdir(root_folder)
if not files[1].endswith('.mp3'):
    pass

for file_name in files:
    
    abs_location = '%s/%s' % (root_folder, file_name)
    
    song_info = eyed3.load(abs_location)
    if song_info is None:
        print('Skipping %s' % abs_location)
        continue
    
    print(song_info.tag.artist)