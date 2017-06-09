import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from keras import backend as K
def reset_session():
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


import numpy as np
import pickle as pkl
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

max_sequence_length = 30
if os.path.exists('tmp/embedding_matrix.npy'):
    embedding_matrix = np.load('tmp/embedding_matrix.npy')
def build_model():
    reset_session()
    dropout_rate=0.2

    weight_initializer = K_init.get('normal', scale=0.1)
    def siamese_conv(pretrain=False):
        input = Input(shape=(max_sequence_length, ), dtype='int32')
        input_mask = Input(shape=(max_sequence_length,), dtype='bool')
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
                                    init=weight_initializer,
                                    mask_zero=False,
                                    )(input)
        cnn_config = [(32, 2), (32, 3), (64,4) , (64, 5), (128, 7)]
        cnn_output = []
        for fs, fl in cnn_config:
            o1 = Conv1D(fs, fl, activation='relu', border_mode='same')(embedding_input)
            o1 = MaxPooling1D(pool_length=30, border_mode='valid')(o1)
            o1 = Flatten()(o1)
            cnn_output.append(o1)
        output = Merge(mode='concat', concat_axis=-1)(cnn_output)
        output = Dense(128, activation='tanh')(output)
        model = Model(input=[input, input_mask], output=output)
        return model
    sen_model = siamese_conv(pretrain=False)
    sen_1 = Input(shape=(max_sequence_length,), dtype='int32')
    sen_1_mask = Input(shape=(max_sequence_length,), dtype='bool')

    sen_2 = Input(shape=(max_sequence_length,), dtype='int32')
    sen_2_mask = Input(shape=(max_sequence_length,), dtype='bool')

    embedding_sen_1 = sen_model([sen_1, sen_1_mask])
    embedding_sen_2 = sen_model([sen_2, sen_2_mask])
    dense_dim = 300
    abs_merge = lambda x: tf.abs(x[0] - x[1])
    mul_merge = lambda x: tf.mul(x[0], x[1])
    abs_feature = Merge(mode=abs_merge, output_shape=lambda x: x[0])([embedding_sen_1, embedding_sen_2])
    mul_feature = Merge(mode=mul_merge, output_shape=lambda x: x[0])([embedding_sen_1, embedding_sen_2])
    leaks_input = Input(shape=(3,), dtype='float32')
    leaks_dense = Dense(50, activation='relu')(leaks_input)

    feature = Merge(mode='concat', concat_axis=-1)([abs_feature, mul_feature, leaks_dense])
    feature = Dropout(dropout_rate)(feature)
    feature = Dense(64, activation='relu')(feature)
    feature = Dropout(dropout_rate)(feature)
    feature = Dense(1, activation='sigmoid')(feature)
    final_model = Model(input=[sen_1, sen_1_mask, sen_2, sen_2_mask, leaks_input], output=feature)
    optimizer = Adam(lr=1e-3)
    final_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return final_model


# train data
X_q1 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q1_clean.npy", mmap_mode='r')
X_q2 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q2_clean.npy", mmap_mode='r')


y_label = np.load("/home/huangzhengjie/quora_pair/quora_stack/label.npy", mmap_mode="r")

# leak_data
if os.path.exists('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/leak_data.npy'):
    train_leaks = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/leak_data.npy', mmap_mode='r')
    test_leaks = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/leak_data.npy', mmap_mode='r')
else:
    train_leaks = pd.read_csv("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/handcraft_feat.csv")
    train_leaks = train_leaks[['interset_count', 'log_max_q1_q2_freq', 'log_min_q1_q2_freq']].as_matrix()
    test_leaks = pd.read_csv("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/handcraft_feat.csv")
    test_leaks = test_leaks[['interset_count', 'log_max_q1_q2_freq', 'log_min_q1_q2_freq']].as_matrix()

    ss = StandardScaler()
    ss.fit(train_leaks)
    train_leaks = ss.transform(train_leaks)
    test_leaks = ss.transform(test_leaks)
    np.save('stage0_output/train/leak_data.npy', train_leaks)
    np.save('stage0_output/test/leak_data.npy', test_leaks)



class hash_deepnet(object):
    def __init__(self):
        self.name = self.__class__.__name__

    def reset(self):
        print 'RESET STATE'
        self.final_model= build_model()

    def train(self, train_index, test_index):
        self.reset()
        train_q1 = X_q1[train_index][:,:max_sequence_length]
        train_q2 = X_q2[train_index][:,:max_sequence_length]
        train_q1_mask = train_q1 > 0
        train_q2_mask = train_q2 > 0
        train_y = y_label[train_index]
        train_l = train_leaks[train_index, :]

        val_q1 = X_q1[test_index][:,:max_sequence_length]
        val_q2 = X_q2[test_index][:,:max_sequence_length]

        val_q1_mask = val_q1 > 0
        val_q2_mask = val_q2 > 0

        val_y = y_label[test_index]
        val_l = train_leaks[test_index, :]
        self.final_model.fit([train_q1, train_q1_mask, train_q2, train_q2_mask, train_l ],
                        train_y,
                        validation_data=([val_q1, val_q1_mask, val_q2, val_q2_mask, val_l ], val_y),
                        batch_size=256,
                        nb_epoch=2)
        val_prediction = self.final_model.predict([val_q1, val_q1_mask, val_q2, val_q2_mask, val_l ], batch_size=512, verbose=1)
        score = log_loss(val_y, np.array(val_prediction, "float64"))
        return val_prediction, score

    def train_all(self):
        self.reset()
        train_q1 = X_q1[:,:max_sequence_length]
        train_q2 = X_q2[:,:max_sequence_length]
        train_q1_mask = train_q1 > 0
        train_q2_mask = train_q2 > 0
        train_y = y_label
        train_l = train_leaks

        self.final_model.fit([train_q1, train_q1_mask, train_q2, train_q2_mask, train_l],
                        train_y,
                        batch_size=256,
                        nb_epoch=2)

        return


    def test(self):
        test_q1 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q1_clean.npy')[:,:max_sequence_length]
        test_q2 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q2_clean.npy')[:,:max_sequence_length]
        test_q1_mask = test_q1 > 0#, test_q1 == embedding_matrix.shape[0] - 1)
        test_q2_mask = test_q2 > 0#, test_q2 == embedding_matrix.shape[0] - 1)
        test_l = test_leaks
        prediction = self.final_model.predict([test_q1, test_q1_mask, test_q2, test_q2_mask, test_l], batch_size=512, verbose=1)
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
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/leak_cnn_clean_%s.npy" % state, X_prediction)
    model.train_all()
    pred = model.test()
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/test/leak_cnn_clean_%s.npy" % state, pred)




