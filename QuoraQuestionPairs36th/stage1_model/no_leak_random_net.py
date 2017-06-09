import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
import numpy as np
import pickle as pkl
import argparse
import keras.initializations as K_init
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler




from sklearn.metrics import log_loss, accuracy_score
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

import sys
sys.path.append('/home/huangzhengjie/quora_pair/')
from birnn import MaxPoolingOverTime, TimeReverse, DotProductLayer, \
        MaskBilinear, MaskMeanPoolingOverTime, MaskSumPoolingOverTime, \
        ElementWiseConcat, StopGradientLayer

def parse_args():
    description = '''
    train_dup am input_left input_right
    '''
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('train_dup', type=int)
    parser.add_argument('am', type=int)
    parser.add_argument('input_left', type=str)
    parser.add_argument('input_right', type=str)
    args = parser.parse_args()
    return args

args = parse_args()
param_grid = {
        "train_dup" : args.train_dup,
        "input_left": args.input_left,
        "input_right": args.input_right,
        "am": args.am
}
if param_grid["input_left"] == param_grid["input_right"]:
    param_grid["syn"] = True
    param_grid["nb_epoch"] = 1
else:
    param_grid["syn"] = False
    param_grid["nb_epoch"] = 1
print param_grid


max_sequence_length = 30
dropout_rate=0.5
if os.path.exists('tmp/embedding_matrix.npy'):
    embedding_matrix = np.load('tmp/embedding_matrix.npy')

weight_initializer = K_init.get('normal', scale=0.1)

def siamese_conv(pretrain=False):
    input = Input(shape=(max_sequence_length, ), dtype='int32')
    embedding_dim = 300
    with tf.device('/cpu:0'):
        if pretrain:
            embedding_input = Embedding(embedding_matrix.shape[0],
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    trainable=True,
                                    mask_zero=False,
                                    )(input)
        else:
            embedding_input = Embedding(embedding_matrix.shape[0],
                                    embedding_dim,
                                    trainable=True,
                                    mask_zero=False,
                                    )(input)
    cnn_output = []
    cnn_config = [(32, 2), (32, 3), (64,4) , (64, 5), (128, 7)]
    for fs, fl in cnn_config:
        o1 = Conv1D(fs, fl, activation='relu', border_mode='same')(embedding_input)
        o1 = MaxPooling1D(pool_length=30, border_mode='valid')(o1)
        o1 = Flatten()(o1)
        cnn_output.append(o1)
    output = Merge(mode='concat', concat_axis=-1)(cnn_output)
    #output = LSTM(128)(embedding_input)
    model = Model(input=[input], output=output)
    return model


sen_1 = Input(shape=(max_sequence_length,), dtype='int32')
sen_2 = Input(shape=(max_sequence_length,), dtype='int32')

sen_model = siamese_conv(pretrain=False)
embedding_sen_1 = sen_model(sen_1)
if param_grid["syn"] is not True:
    sen2_model = siamese_conv(pretrain=False)
    embedding_sen_2 = sen2_model(sen_2)
else:
    embedding_sen_2 = sen_model(sen_2)
dense_dim = 300

if param_grid['am'] == 1:
    abs_merge = lambda x: tf.abs(x[0] - x[1])
    mul_merge = lambda x  : tf.mul(x[0], x[1])
    abs_feature = Merge(mode=abs_merge, output_shape=lambda x: x[0])([embedding_sen_1, embedding_sen_2])
    mul_feature = Merge(mode=mul_merge, output_shape=lambda x: x[0])([embedding_sen_1, embedding_sen_2])
    feature = Merge(mode='concat', concat_axis=-1)([abs_feature, mul_feature])
else:
    feature = Merge(mode='concat', concat_axis=-1)([embedding_sen_1, embedding_sen_2])
feature = Activation('relu')(feature)
feature = Dense(1, activation='sigmoid')(feature)
final_model = Model(input=[sen_1, sen_2], output=feature)


optimizer = Nadam(lr=2e-3)
#optimizer = Adam(lr=1e-2)
final_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# train data
X_q1 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q1%s.npy" % param_grid['input_left'], mmap_mode='r')
X_q2 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q2%s.npy" % param_grid['input_left'], mmap_mode='r')

X_q1_2 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q1%s.npy" % param_grid['input_right'], mmap_mode='r')
X_q2_2 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q2%s.npy" % param_grid['input_right'], mmap_mode='r')


x_flag = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/no_leak_flag.npy")

y_label = np.load("/home/huangzhengjie/quora_pair/quora_stack/label.npy", mmap_mode="r")


class hash_deepnet(object):
    def __init__(self):
        print 'Compiling model'
        self.name = self.__class__.__name__
        self.init_weights = final_model.get_weights()
        self.init_optimizer = final_model.optimizer.get_weights()

    def reset(self):
        print 'RESET STATE'
        final_model.set_weights(self.init_weights)
        final_model.optimizer.set_weights(self.init_optimizer)

    def train(self, train_index, test_index):
        self.reset()
        train_q1 = X_q1[train_index][:,:max_sequence_length]
        train_q2 = X_q2[train_index][:,:max_sequence_length]
        train_q1_2 = X_q1_2[train_index][:,:max_sequence_length]
        train_q2_2 = X_q2_2[train_index][:,:max_sequence_length]
        train_y = y_label[train_index]
        train_data_q1 = np.vstack([train_q1, train_q2])
        train_data_q2 = np.vstack([train_q2_2, train_q1_2])
        train_data_y = np.concatenate([train_y, train_y])
        if param_grid['train_dup'] == 1:
            train_flag = x_flag[train_index]
            train_flag = np.concatenate([train_flag, train_flag])
            train_data_q1 = train_data_q1[train_flag]
            train_data_q2 = train_data_q2[train_flag]
            train_data_y = train_data_y[train_flag]




        val_q1 = X_q1[test_index][:,:max_sequence_length]
        val_q2 = X_q2[test_index][:,:max_sequence_length]
        val_q1_2 = X_q1_2[test_index][:,:max_sequence_length]
        val_q2_2 = X_q2_2[test_index][:,:max_sequence_length]
        val_y = y_label[test_index]

        val_data_q1 = np.vstack([val_q1, val_q2])
        val_data_q2 = np.vstack([val_q2_2, val_q1_2])
        val_data_y = np.concatenate([val_y, val_y])
        if param_grid['train_dup'] == 1:
            val_flag = x_flag[test_index]
            val_flag = np.concatenate([val_flag, val_flag])
            val_data_q1 = val_data_q1[val_flag]
            val_data_q2 = val_data_q2[val_flag]
            val_data_y = val_data_y[val_flag]



        final_model.fit([train_data_q1, train_data_q2],
                        train_data_y,
                        validation_data=([val_data_q1, val_data_q2], val_data_y),
                        batch_size=256,
                        nb_epoch=param_grid['nb_epoch'])
        val_prediction = final_model.predict([val_q1, val_q2_2], batch_size=8192, verbose=1)
        val_prediction += final_model.predict([val_q2, val_q1_2], batch_size=8192, verbose=1)
        val_prediction /= 2
        score = log_loss(val_y, np.array(val_prediction, "float64"))
        print score
        return val_prediction, score

    def train_all(self):
        self.reset()
        train_q1 = X_q1[:,:max_sequence_length]
        train_q2 = X_q2[:,:max_sequence_length]
        train_q1_2 = X_q1_2[:,:max_sequence_length]
        train_q2_2 = X_q2_2[:,:max_sequence_length]
        train_y = y_label
        train_data_q1 = np.vstack([train_q1, train_q2])
        train_data_q2 = np.vstack([train_q2_2, train_q1_2])
        train_data_y = np.concatenate([train_y, train_y])
        if param_grid['train_dup'] == 1:
            train_flag = x_flag[train_index]
            train_flag = np.concatenate([train_flag, train_flag])
            train_data_q1 = train_data_q1[train_flag]
            train_data_q2 = train_data_q2[train_flag]
            train_data_y = train_data_y[train_flag]

        final_model.fit([train_data_q1, train_data_q2],
                        train_data_y,
                        batch_size=256,
                        nb_epoch=param_grid['nb_epoch'])
        return


    def test(self):
        test_q1 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q1%s.npy' % param_grid['input_left'])[:,:max_sequence_length]
        test_q2 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q2%s.npy' % param_grid['input_left'])[:,:max_sequence_length]
        test_q1_2 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q1%s.npy' % param_grid['input_right'])[:,:max_sequence_length]
        test_q2_2 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q2%s.npy' % param_grid['input_right'])[:,:max_sequence_length]
        prediction = final_model.predict([test_q1, test_q2_2], batch_size=8192, verbose=1)
        prediction += final_model.predict([test_q2, test_q1_2], batch_size=8192, verbose=1)
        prediction /= 2
        return prediction


if __name__ == "__main__":
    model = hash_deepnet()
    with open("/home/huangzhengjie/quora_pair/quora_stack/kdf.pkl",'r') as f:
        kfd = pkl.load(f)

    X_prediction = np.zeros((X_q1.shape[0], 1), dtype='float32')
    score_log = []
    for fold_id, (train_index, test_index) in enumerate(kfd):
        print 'Fold', fold_id
        target_output, score = model.train(train_index, test_index)
        X_prediction[test_index] = target_output
        print 'Fold', fold_id, 'log_loss:', score
        score_log.append(score)

    state = "%s_%s" % (np.average(score_log), np.std(score_log))
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/no_leak_cnn_am%s_d_%s_l%s_r%s_%s.npy" % (param_grid['am'], param_grid['train_dup'], param_grid['input_left'], param_grid['input_right'], state), X_prediction)
    model.train_all()
    pred = model.test()
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/test/no_leak_cnn_am%s_d_%s_l%s_r%s_%s.npy" % (param_grid['am'], param_grid['train_dup'], param_grid['input_left'], param_grid['input_right'], state), pred)




