import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
import pickle as pkl
import sys
import pandas as pd
import json
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# ensemble different models

from keras.models import Model, Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape,\
                        Merge, BatchNormalization, TimeDistributed,\
                        Lambda, Activation, LSTM, Flatten,\
                        Conv1D, GRU, MaxPooling1D,\
                        Conv2D, Input, MaxPooling2D, BatchNormalization, Masking

from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.optimizers import SGD, Adam, Nadam
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from sklearn.metrics import roc_curve, roc_auc_score





leakage_name = []
feat_file = "all_feat"
with open("/home/huangzhengjie/quora_pair/quora_stack/tmp/feat_name/" + feat_file, 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        leakage_name.append(line.strip())




# build train_features
train_leakage_feature = pd.read_csv('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/handcraft_feat.csv')
train_leakage_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
train_leakage_feature.fillna(0, inplace=True)

train_leakage_feature = train_leakage_feature.as_matrix(columns=leakage_name)

test_leakage_feature = pd.read_csv('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/handcraft_feat.csv')
test_leakage_feature.replace([np.inf, -np.inf], np.nan, inplace=True)
test_leakage_feature.fillna(0, inplace=True)

test_leakage_feature = test_leakage_feature.as_matrix(columns=leakage_name)



print 'Starting Normalization'
sys.stdout.flush()
std = StandardScaler()
std.fit(np.vstack([train_leakage_feature, test_leakage_feature]))
train_leakage_feature = std.transform(train_leakage_feature)
test_leakage_feature = std.transform(test_leakage_feature)

# Define Keras Model
dropout_rate = 0.3
dense_dim = 64
num_layer = 2
print 'Param'
print 'Dropout', dropout_rate
print 'Dense Dim', dense_dim
print 'Num layer', num_layer
model = Sequential()
model.add(Dense(dense_dim, input_dim=train_leakage_feature.shape[1]))
model.add(BatchNormalization())
model.add(Activation('relu'))
for _ in range(np.random.choice(num_layer)):
    model.add(Dense(dense_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = Nadam(lr=0.05)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
init_weights = model.get_weights()
init_optimizer = model.optimizer.get_weights()





x_train = train_leakage_feature
x_test = test_leakage_feature

y_train = np.load("/home/huangzhengjie/quora_pair/quora_stack/label.npy", mmap_mode="r")




with open("/home/huangzhengjie/quora_pair/quora_stack/kdf.pkl", 'r') as f:
    kfd = pkl.load(f)

xgb_train_output = np.zeros(len(x_train))
score_log = []
for fold_id, (train_index, test_index) in enumerate(kfd):
    print 'Fold', fold_id
    model.set_weights(init_weights)
    model.optimizer.set_weights(init_optimizer)
    model.fit(x_train[train_index], y_train[train_index],\
            validation_data=(x_train[test_index], y_train[test_index]), \
            batch_size=128,
            nb_epoch=10,
            shuffle=True)
    val_pred = model.predict(x_train[test_index], batch_size=8192, verbose=1)
    score = log_loss(y_train[test_index], np.array(val_pred, dtype='float64'))
    print 'Fold', fold_id, 'log_loss:', score
    score_log.append(score)

    xgb_train_output[test_index] = val_pred
state = "%s_%s" % (np.average(score_log), np.std(score_log))
np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/nn_models_%s.npy" % state, xgb_train_output)
# train_all
model.set_weights(init_weights)
model.optimizer.set_weights(init_optimizer)
model.fit(x_train, y_train,\
        batch_size=128,
        nb_epoch=10,
        shuffle=True)
pred = model.predict(x_test, batch_size=8192, verbose=1)
np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/test/nn_models_%s.npy" % state, pred)
