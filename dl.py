from numpy import array, zeros, log, exp
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Conv2D, AtrousConvolution2D, Flatten, Dense, MaxPooling2D, Dropout, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import backend as K

data_dir = "data/set1/"
data_type = {"Filename": str, "Genres": str, "Release Year": int}
albums = pd.read_csv(data_dir + "albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])

# Label to index number
token = Tokenizer()
genres = list(albums.Genres.get_values())
token.fit_on_texts(genres)

idx_word = dict()
for key in token.word_index.keys():
    idx_word[token.word_index[key]] = key

label_lst = albums.Genres.get_values()
data_y = zeros((len(label_lst), max(token.word_index.values())+1))

# Dummy the labels (which means that model has as many as number of all genres)
i=0
for album_labels in label_lst:
    splt_labels = album_labels.split()
    for label in splt_labels:
        data_y[i, token.word_index[label]] = 1
    i += 1

# Read image
X_origin = array([array(Image.open(data_dir+"resize/"+filename+".jpg")) for filename in albums.Filename.get_values()])
X = zeros((X_origin.shape[0], 128, 128, 3))
for i in xrange(X_origin.shape[0]):
    X[i] = X_origin[i]

# Normalization
Xmean = np.mean(X, axis=0)
Xstd  = np.std(X, axis=0)
X -= Xmean
X /= (Xstd+0.0001)

# compute loss weight for each genres cuz training data are unbalanced on genres
data_y = data_y[:, 2:]
cate_sum = np.sum(data_y, axis=0) # cate_sum is a decreasing sequence
for i in xrange(len(cate_sum)):
    if cate_sum[i]<10: # the criteria is to get rid of the genres with less than 10 training data
        break
data_y = data_y[:,:i]
cate_weight = -log(1.-1./cate_sum[:i])
cate_weight = 10**(10*cate_weight)

# split data
train_x, test_x, train_y, test_y = train_test_split(X, data_y, test_size=0.2)

# Build NN Model (model to optimize)
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=train_x.shape[1:], dim_ordering='tf'))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1], activation='sigmoid'))

#def f(x):
#    return x / (K.max(x))
#model.add(Activation(f))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          validation_data = (test_x, test_y),
          batch_size=5,
          nb_epoch=10,
          class_weight=cate_weight)


def greater50percent(y):
    p = []
    for s in y:
        tmp = []
        for i in xrange(len(s)):
            if s[i] > 0.5:
                tmp.append(i)
        p.append(tmp)
    return p

pred = model.predict(test_x)
p  = greater50percent(pred)
p_ = greater50percent(test_y)

for i in xrange(len(p)):
    print(i, p[i], p_[i])

def predict_req(filename, model):
    reqdata = array([array(Image.open(filename))])
    return model.predict(reqdata)
