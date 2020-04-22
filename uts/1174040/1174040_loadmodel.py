# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:39:46 2020

@author: User
"""

from keras.models import load_model
import numpy as np
import glob
import librosa
import librosa.feature


model = load_model('penyanyi.h5')

#%%
feature = set_features_and_labels('singer/testing.wav')
model.predict_classes(feature)