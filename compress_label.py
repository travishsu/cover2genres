from numpy import array, zeros, log, exp
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Input, ZeroPadding2D, Conv2D, MaxPooling2D, Dropout, Flatten

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

def create_autoencoder(train_iteration=10000):
    encoded_dim = 32

    label = Input(shape=(data_y.shape[1],))
    encoded = Dense(80, activation='relu')(label)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(48, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(48, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(80, activation='relu')(decoded)
    decoded = Dense(data_y.shape[1], activation='sigmoid')(decoded)

    autoencoder = Model(input=label, output=decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='sgd')
    autoencoder.fit(data_y, data_y, batch_size=32, nb_epoch=train_iteration)

    compressed = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-4](compressed)
    decoder_layer = autoencoder.layers[-3](decoder_layer)
    decoder_layer = autoencoder.layers[-2](decoder_layer)
    decoder_layer = autoencoder.layers[-1](decoder_layer)
    decoder = Model(input=compressed, output=decoder_layer)

    encoder = Model(input=label, output=encoded)
    compressed = Input(shape=(32,))
    decoder_layer = autoencoder.layers[-4](compressed)
    decoder_layer = autoencoder.layers[-3](decoder_layer)
    decoder_layer = autoencoder.layers[-2](decoder_layer)
    decoder_layer = autoencoder.layers[-1](decoder_layer)
    decoder = Model(input=compressed, output=decoder_layer)

    return encoder, decoder

encoder, decoder = create_autoencoder()
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

# split data
encoded = encoder.predict(data_y)
train_x, test_x, train_y, test_y = train_test_split(X, encoded, test_size=0.2)

# Build NN Model (model to optimize)
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=train_x.shape[1:], dim_ordering='tf'))
model.add(Conv2D(32, 3, 3, border_mode='valid', activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, 3, 3, border_mode='valid', activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Flatten())
#model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1]))

#def f(x):
#    return x / (K.max(x))
#model.add(Activation(f))

model.compile(loss='mse',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          validation_data = (test_x, test_y),
          batch_size=32,
          nb_epoch=10)

def greater50percent(y):
    p = []
    for s in y:
        tmp = []
        for i in xrange(len(s)):
            if s[i] > 0.5:
                tmp.append(i)
        p.append(tmp)
    return p

pred = decoder.predict(model.predict(test_x))
p  = greater50percent(pred)
p_ = greater50percent(test_y)

for i in xrange(len(p)):
    print(i, p[i], p_[i])
