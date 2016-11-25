from keras.preprocessing.text import Tokenizer
from numpy import array, zeros
from PIL import Image
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, AtrousConvolution2D, Flatten, Dense, Activation, MaxPooling2D
from keras.layers import Dropout
from keras.optimizers import SGD
from keras import backend as K

data_type = {"Filename": str, "Genres": str, "Release Year": int}
albums = pd.read_csv("albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])
albums = albums.sample(130)

token = Tokenizer()
genres = list(albums.Genres.get_values())
token.fit_on_texts(genres)

label_lst = albums.Genres.get_values()
train_y = zeros((len(label_lst), max(token.word_index.values())+1))

i=0
for album_labels in label_lst:
    splt_labels = album_labels.split()
    for label in splt_labels:
        train_y[i, token.word_index[label]] = 1
    i += 1

train_x = array([array(Image.open("data/"+filename+".jpg")) for filename in albums.Filename.get_values()])
X = zeros((train_x.shape[0], 300, 300, 3))
for i in xrange(train_x.shape[0]):
    X[i] = train_x[i]

X /= 255
Xmean = np.mean(X, axis=0)
Xstd  = np.std(X, axis=0)

X -= Xmean
X /= (Xstd+0.0001)

# Build NN Model
model = Sequential()
model.add(Conv2D(16, 3, 3, border_mode='same', input_shape=(300,300,3), dim_ordering='tf'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D((4, 4)))
model.add(Conv2D(16, 3, 3, border_mode='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, 3, 3, border_mode='same'))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(32, 5, 5, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Conv2D(32, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1]))
#model.add(Activation('softmax'))
model.add(Activation('sigmoid'))

#def f(x):
#    return x / (K.max(x))
#model.add(Activation(f))

#sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy',
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X[:100], train_y[:100],
          batch_size=5,
          nb_epoch=100)

def greater10percent(y):
    p = []
    for s in y:
        tmp = []
        for i in xrange(len(s)):
            if s[i] > 0.5:
                tmp.append(i)
        p.append(tmp)
    return p

pred = model.predict(X)
p  = greater10percent(pred)
p_ = greater10percent(train_y)

idx_word = dict()
for key in token.word_index.keys():
    idx_word[token.word_index[key]] = key

for i in xrange(len(p)):
    print(i, p[i], p_[i])

def predict_req(filename, model):
    reqdata = array([array(Image.open(filename))])
    return model.predict(reqdata)
