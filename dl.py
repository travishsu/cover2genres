from numpy import array, zeros, log, exp
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Conv2D, AtrousConvolution2D, Flatten, Dense, Activation, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import backend as K

data_type = {"Filename": str, "Genres": str, "Release Year": int}
albums = pd.read_csv("albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])

token = Tokenizer()
genres = list(albums.Genres.get_values())
token.fit_on_texts(genres)

idx_word = dict()
for key in token.word_index.keys():
    idx_word[token.word_index[key]] = key

label_lst = albums.Genres.get_values()
data_y = zeros((len(label_lst), max(token.word_index.values())+1))

i=0
for album_labels in label_lst:
    splt_labels = album_labels.split()
    for label in splt_labels:
        data_y[i, token.word_index[label]] = 1
    i += 1

X_origin = array([array(Image.open("data/"+filename+".jpg")) for filename in albums.Filename.get_values()])
X = zeros((X_origin.shape[0], 300, 300, 3))
for i in xrange(X_origin.shape[0]):
    X[i] = X_origin[i]

X /= 255
Xmean = np.mean(X, axis=0)
Xstd  = np.std(X, axis=0)

X -= Xmean
X /= (Xstd+0.0001)

data_y = data_y[:, 2:]
cate_sum = np.sum(data_y, axis=0)
for i in xrange(len(cate_sum)):
    if cate_sum[i]<10:
        break
cate_weight = -log(1.-1./cate_sum[:i])
cate_weight = 10**(10*cate_weight)
data_y = data_y[:,:i]

train_x, test_x, train_y, test_y = train_test_split(X, data_y, test_size=0.2)

# Build NN Model
model = Sequential()
model.add(Conv2D(32*2, 3, 3, border_mode='same', input_shape=(300,300,3), dim_ordering='tf'))
model.add(Activation('relu'))
model.add(MaxPooling2D((4, 4)))
#model.add(BatchNormalization())
model.add(Conv2D(40*2, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
#model.add(BatchNormalization())
model.add(Conv2D(48*2, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

#model.add(Conv2D(32, 5, 5, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(Conv2D(32, 3, 3, border_mode='valid'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1]))
#model.add(Activation('softmax'))
model.add(Activation('tanh'))

#def f(x):
#    return x / (K.max(x))
#model.add(Activation(f))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy',
#model.compile(loss='binary_crossentropy',
#              optimizer=sgd,
model.compile(loss='hinge',
	      optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          validation_data = (test_x, test_y),
          batch_size=5,
          nb_epoch=10,
          class_weight=cate_weight)

def greater10percent(y):
    p = []
    for s in y:
        tmp = []
        for i in xrange(len(s)):
            if s[i] > 0.1:
                tmp.append(i)
        p.append(tmp)
    return p

pred = model.predict(test_x)
p  = greater10percent(pred)
p_ = greater10percent(test_y)

for i in xrange(len(p)):
    print(i, p[i], p_[i])

def predict_req(filename, model):
    reqdata = array([array(Image.open(filename))])
    return model.predict(reqdata)
