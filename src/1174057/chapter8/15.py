# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:30:54 2020

@author: Alit
"""

    print("Loading data...")
    volumes = get3DImages(data_dir=data_dir)
    volumes = volumes[..., np.newaxis].astype(np.float)
    print("Data loaded...")