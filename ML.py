# Load libraries
import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import Callback
from keras import backend as K
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

def load(file, nrows):
    df = pd.read_csv(file, nrows=nrows)

    # convert the column values from literal string to dictionary
    df['ltiFeatures'] = df['ltiFeatures'].apply(ast.literal_eval)
    df['stiFeatures'] = df['stiFeatures'].apply(ast.literal_eval)
   
    return df

# load all the data
training = load("training.csv", nrows=None)
interest_topics = pd.read_csv("interest_topics.csv")

print("Finished loading training data...")

CategoryID = interest_topics.values[:, 0]

X_train = []
Y_train = []
for person in training.values:
    subarr = []
    for i in range(len(CategoryID)):
        subarr.append(person[2].get(str(CategoryID[i]), 0))
    X_train.append(subarr)
    Y_train.append(person[1])

X_train = np.asarray(X_train)
Y_train = np.asarray(Y_train)
X_train, Y_train = shuffle(X_train, Y_train)

d = {}
for i in range(len(CategoryID)):
   lst = interest_topics.values[:, 1][i].split('/')
   if lst[1] not in d:
      d[lst[1]] = []
   d[lst[1]].append(i)

X_modtrain = []

for entry in X_train:
   slst = []
   for key in d.keys():
      sum = 0
      for j in range(len(d[key])):
         sum += entry[d[key][j]]
      slst.append(sum)
   X_modtrain.append(slst)

X_train = np.asarray(X_modtrain)

X_train, validation_data, Y_train, validation_target = train_test_split(X_train,
                                                                    Y_train,
                                                                    test_size=0.25)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f' — val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}')
        return

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

metrics = Metrics()

#Multilayer Perceptron for Binary Classification
model = Sequential()
model.add(Dense(8, input_dim=len(d), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=f1_loss,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=128,
          validation_data=(validation_data, validation_target),
          callbacks=[metrics], verbose=2)
print("Finished training data...")

validation = load("validation.csv", None)
print("Finished loading validation data...")

#evaluation on validation dataset

X_eval = []
Y_eval = []
for person in validation.values:
    subarr = []
    for i in range(len(CategoryID)):
        subarr.append(person[2].get(str(CategoryID[i]), 0))
    X_eval.append(subarr)
    Y_eval.append(person[1])

X_eval = np.asarray(X_eval)
Y_eval = np.asarray(Y_eval)

d = {}
for i in range(len(CategoryID)):
   lst = interest_topics.values[:, 1][i].split('/')
   if lst[1] not in d:
      d[lst[1]] = []
   d[lst[1]].append(i)

X_modtrain = []

for entry in X_eval:
   slst = []
   for key in d.keys():
      sum = 0
      for j in range(len(d[key])):
         sum += entry[d[key][j]]
      slst.append(sum)
   X_modtrain.append(slst)

X_eval = np.asarray(X_modtrain)

_, accuracy = model.evaluate(X_eval, Y_eval, verbose=2)
print('Final Validation Accuracy: %.2f' % (accuracy*100))
#Achiev
