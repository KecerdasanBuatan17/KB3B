# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:26:02 2020

@author: TIKOMDIK 01
"""
from keras.models import load_model
import numpy as np
import glob
import librosa
import librosa.feature

model = load_model('model_lagu.h5')
def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

def set_features_and_labels(file):
    all_features = []
    sound_files = glob.glob(file)
    for f in sound_files:
        features = extract_features_song(f)
        all_features.append(features)

    return np.stack(all_features)
#%%
feature = set_features_and_labels('D:/Kuliah/Python/Kecerdasan_Buatan/Project_AI/sample6/UTS/11.wav')
model.predict(feature)
model.predict_classes(feature)