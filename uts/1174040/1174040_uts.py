# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:49:48 2020

@author: User
"""
import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
#%%
def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]
    
#%%
def generate_features_and_labels():
    all_features = []
    all_labels = []

    singers = ['back number', 'ikimonogakari', 'kylee', 'monkey magik', 'sammy simorangkir', 'scandal', 'shoko nakagawa', 'sid', 'tomohisa sako', 'wacci']
    for singer in singers:
        sound_files = glob.glob('singer/'+singer+'/*.wav')
        print('Processing %d songs in %s singer...' % (len(sound_files), singer))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(singer)

    # convert labels to one-hot encoding cth blues : 1000000000 classic 0100000000
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)#ke integer
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))#ke one hot
    return np.stack(all_features), onehot_labels    
    
 
#%%    
features, labels = generate_features_and_labels()
#%%
def set_features_and_labels(file):
    all_features = []

    
    sound_files = glob.glob(file)
#    print('Processing %d songs in %s genre...' % (len(sound_files), genre))
    for f in sound_files:
        features = extract_features_song(f)
        all_features.append(features)

    return np.stack(all_features)
feature = set_features_and_labels('singer/kaulah segalanya.wav')
model.predict_classes(feature)  
#%%
model.save('penyanyi.h5')
