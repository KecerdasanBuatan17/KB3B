# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:53:44 2020

@author: Alit
"""

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

def display_mfcc(lagu):
    y, _ = librosa.load(lagu)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(lagu)
    plt.tight_layout()
    plt.show()

display_mfcc('C:/Users/Alit/Desktop/1174057/song/music/1/blues.00005.wav')

display_mfcc('C:/Users/Alit/Desktop/1174057/song/music/2/classical.00005.wav')

def extract_features_lagu(f):
    y, _ = librosa.load(f)
    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[-25000:]

extract_features_lagu('C:/Users/Alit/Desktop/1174057/song/music/1/blues.00005.wav')

extract_features_lagu('C:/Users/Alit/Desktop/1174057/song/music/2/classical.00005.wav')

extract_features_lagu('C:/Users/Alit/Desktop/1174057/song/music/3/country.00005.wav')

def generate_features_and_labels():
    all_features = []
    all_labels = []

    music = ['1', '2' , '3', '4', '5', '6', '7', '8', '9', '10']
    for song in music:
        sound_files = glob.glob('C:/Users/Alit/Desktop/1174057/song/music/'+song+'/*.wav')
        print('Processing %d lagu by %s ...' % (len(sound_files), song))
        
        for f in sound_files:
            features = extract_features_lagu(f)
            all_features.append(features)
            all_labels.append(song)
    
    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    print(np.stack(all_features), onehot_labels)
    return np.stack(all_features), onehot_labels

features, labels = generate_features_and_labels()
    
print(np.shape(features))
print(np.shape(labels))

training_split = 0.8

# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

print(np.shape(train))
print(np.shape(test))

train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]

print(np.shape(train_input))
print(np.shape(train_labels))

print(np.shape(test_input))
print(np.shape(test_labels))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100, input_dim=np.shape(train_input)[1]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# save the trained model
model.save("lagu2.hdf5")

import tensorflow as tf 
model2 = tf.keras.models.load_model("lagu2.hdf5")
print(model2.summary())

def predict(lagu_path):
    lagu = np.stack([extract_features_lagu(lagu_path)])
    # do the prediction
    prediction = model2.predict(lagu, batch_size=32)

    print("Prediction: %s, confidence: %.2f" % (np.argmax(prediction), np.max(prediction)))

predict('C:/Users/Alit/Desktop/1174057/song/music/2/classical.00005.wav')

predict('C:/Users/Alit/Desktop/1174057/song/music/1/blues.00005.wav')

predict('C:/Users/Alit/Desktop/1174057/song/music/3/country.00005.wav')

from sklearn.metrics import confusion_matrix
pred_labels = model2.predict(test_input)
cm = confusion_matrix(test_labels.argmax(axis=1), pred_labels.argmax(axis=1))
cm

import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(6,6), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

import numpy as np

music = ['1', '2' , '3', '4', '5', '6', '7', '8', '9', '10']
plot_confusion_matrix(cm, classes=music, normalize=True)
plt.show()