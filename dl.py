from numpy import array, zeros, log, exp
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, AtrousConvolution2D, Flatten, Dense, MaxPooling2D, Dropout, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

data_dir = "data/set2/"
topN     = 12
excludeOther = True

data_type = {"Filename": str, "Genres": str, "Release Year": int}
albums = pd.read_csv(data_dir + "albumlabel.csv", dtype=data_type, parse_dates=["Release Year"])

# Label to index number
token = Tokenizer()
genres = list(albums.Genres.get_values())
token.fit_on_texts(genres)

idx_word = dict()
for key in token.word_index.keys():
    idx_word[token.word_index[key]-1] = key

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
X = zeros((X_origin.shape[0], 32, 32, 3))
for i in xrange(X_origin.shape[0]):
    X[i] = X_origin[i]

# Normalization
Xmean = np.mean(X, axis=0)
Xstd  = np.std(X, axis=0)
X -= Xmean
X /= (Xstd+0.0001)

# compute loss weight for each genres cuz training data are unbalanced on genres
data_y = data_y[:, 1:]
new_index = [idx_word[i] for i in xrange(topN)]
if excludeOther:
    data_y = data_y[:, :topN]
else:
    tmp = np.sum(data_y[:, topN:], axis = 1)
    tmp = tmp / np.maximum(1, tmp)
    data_y = np.concatenate( (data_y[:, :topN] , tmp.astype(int).reshape(data_y.shape[0], 1)), axis = 1)
    new_index.append('other')
cate_sum = np.sum(data_y, axis=0) # cate_sum is a decreasing sequence

label_cardinality = np.mean(cate_sum)
label_density     = label_cardinality/np.sum(cate_sum)
# split data
train_x, test_x, train_y, test_y = train_test_split(X, data_y, test_size=0.2)

# Build NN Model (model to optimize)
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=train_x.shape[1:], dim_ordering='tf'))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(16, 3, 3, border_mode='valid', activation='tanh'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten(input_shape=model.output_shape[1:]))
model.add(BatchNormalization())
model.add(Dense(1024, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(train_y.shape[1], activation='sigmoid'))

#def f(x):
#    return x / (K.max(x))
#model.add(Activation(f))

sgd = SGD(lr=0.01, decay=1e-6, momentum=1, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          #validation_data = (test_x, test_y),
          batch_size=256,
          nb_epoch=100,
          shuffle=True,
          class_weight='auto')


def greater50percent(y):
    p = []
    for s in y:
        tmp = []
        for i in xrange(len(s)):
            if s[i] > 0.5:
                tmp.append(new_index[i])
        p.append(tmp)
    return p

pred = model.predict(test_x)
p  = greater50percent(pred)
p_ = greater50percent(test_y)

# Refer to https://github.com/suraj-deshmukh/Multi-Label-Image-Classification/blob/master/miml.ipynb
threshold = np.arange(0.1,0.9,0.1)
acc = []
accuracies = []
best_threshold = np.zeros(pred.shape[1])
for i in range(pred.shape[1]):
    y_prob = np.array(pred[:,i])
    for j in threshold:
        y_pred = [1 if prob>=j else 0 for prob in y_prob]
        acc.append( matthews_corrcoef(test_y[:,i],y_pred))
    acc   = np.array(acc)
    index = np.where(acc==acc.max())
    accuracies.append(acc.max())
    best_threshold[i] = threshold[index[0][0]]
    acc = []
pred_y = np.array([[1 if pred[i,j]>=best_threshold[j] else 0 for j in range(test_y.shape[1])] for i in range(len(test_y))])

for i in xrange(len(p)):
    print(i, p[i], p_[i])
print(hamming_loss(pred_y, test_y))

def predict_req(filename, model):
    reqdata = array([array(Image.open(filename))])
    return model.predict(reqdata)
