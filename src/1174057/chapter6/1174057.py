# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:59:33 2020

@author: Alit
"""

import librosa #import librosa tang digunakan untuk fungsi mfcc
import librosa.feature #import librosa featuse dan pada baris
import librosa.display #import librosa display
import glob #import glob 
import numpy as np #insert numpy untuk pengolahan data menjadi vektor
import matplotlib.pyplot as plt #import matplotlib untuk melakukan ploting
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

# In[1]: membuat fungsi mfcc untuk melakukan pengujian
def display_mfcc(song): #membuat fungsi mfcc (display_mfcc) dan terdapat variabel Y
    y, _ = librosa.load(song) #method librosa load 
    mfcc = librosa.feature.mfcc(y) # method librosa featurea mfcc

    plt.figure(figsize=(10, 4)) #membuat ﬂot ﬁgure dengan ukuran 10 banding 4
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel') #data librosa display dengan variabel x nya yaitu waktu dan y yaitu mel atau Hz
    plt.colorbar() #plot warna 
    plt.title(song) #plot judul 
    plt.tight_layout()
    plt.show() #plot di tampilkan
    
# In[2]: cek fungsi
display_mfcc('C:/Users/Alit/Desktop/KB3B/genres/hiphop/hiphop.00050.au') 
#untuk mendisplay tampilan glombang suara dari ﬁle 266093 stereo-surgeon kick-loop-5.wav menggunakan metode mfcc
# In[2]: cek fungsi
display_mfcc('C:/Users/Alit/Desktop/KB3B/genres/jazz/jazz.00050.au')
# In[2]: cek fungsi
display_mfcc('C:/Users/Alit/Desktop/KB3B/genres/pop/pop.00050.au')
# In[2]: cek fungsi
display_mfcc('C:/Users/Alit/Desktop/KB3B/genres/reggae/reggae.00050.au')
# In[2]: cek fungsi
display_mfcc('C:/Users/Alit/Desktop/KB3B/genres/rock/rock.00050.au')

# In[3]:
def extract_features_song(f): #nama extract features song yang nantinya akan di gunakan pada fungsi yang lainya
    y, _ = librosa.load(f) #membuat variabel y dengan method librosa load

    mfcc = librosa.feature.mfcc(y) #membuat variabel baru mfcc dengan isi librosa features mfcc dengan isi variabel y

    mfcc /= np.amax(np.absolute(mfcc)) #variabel mfcc dengan isian np.max

    return np.ndarray.flatten(mfcc)[:25000] #membuat array dari data tersebut merupakan data 25000 data pertama

# In[4]:
def generate_features_and_labels(): #pendeﬁnisian nama fungsi yaitu generate features and labels
    all_features = [] #variabel baru dengan array all features
    all_labels = [] #variabel baru dengan array all labels

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'] #kemudian mendeﬁnisikan isian label untuk gendre
    for genre in genres:
        sound_files = glob.glob('genres/'+genre+'/*.au')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding cth blues : 1000000000 classic 0100000000
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)#ke integer
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))#ke one hot
    return np.stack(all_features), onehot_labels

# In[5]: passing parameter dari fitur ekstraksi menggunakan mfcc
features, labels = generate_features_and_labels()

#%%
print(np.shape(features))
print(np.shape(labels))

# In[6] fitur ekstraksi
training_split = 0.8

#%%
# last column has genre, turn it into unique ids
alldata = np.column_stack((features, labels))

#%%
np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

#%%
print(np.shape(train))
print(np.shape(test))

#%%
train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]

#%%
print(np.shape(train_input))
print(np.shape(train_labels))

# In[7]: membuat seq NN, layer pertama dense dari 100 neurons
model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
    
# In[8]: 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

# In[9]:
model.fit(train_input, train_labels, epochs=10, batch_size=32,
          validation_split=0.2)

# In[10]
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

# In[]: 
print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

# In[]: 
model.predict(test_input[:1])
