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
dropout_rate=0.5
if os.path.exists('tmp/embedding_matrix.npy'):
    embedding_matrix = np.load('tmp/embedding_matrix.npy')

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
    # masking
    #embedding_input = LSTM(256, return_sequences=True)(embedding_input)
    def func(x):
        # x[0] input, x[1] mask
        masking = tf.cast(x[1], tf.float32)
        x = x[0]
        masking = tf.tile(tf.expand_dims(masking, -1), [1, 1, 300])
        return tf.mul(x, masking)
    embedding_input = Merge(mode=func, output_shape=lambda x: x[0])([embedding_input, input_mask])
    model = Model(input=[input, input_mask], output=embedding_input)
    return model


sen_model = siamese_conv(pretrain=True)
sen_1 = Input(shape=(max_sequence_length,), dtype='int32')
sen_1_mask = Input(shape=(max_sequence_length,), dtype='bool')

sen_2 = Input(shape=(max_sequence_length,), dtype='int32')
sen_2_mask = Input(shape=(max_sequence_length,), dtype='bool')

embedding_sen_1 = sen_model([sen_1, sen_1_mask])
embedding_sen_2 = sen_model([sen_2, sen_2_mask])

sum_embedding_sen_1 = MaskMeanPoolingOverTime()([embedding_sen_1, sen_1_mask])
sum_embedding_sen_2 = MaskMeanPoolingOverTime()([embedding_sen_2, sen_2_mask])


dense_dim = 300

def gru_model():
    def func(x):
        # x[0] input, x[1] mask
        masking = tf.cast(x[1], tf.float32)
        x = x[0]
        masking = tf.tile(tf.expand_dims(masking, -1), [1, 1, dense_dim])
        return tf.mul(x, masking)
    input = Input(shape=(max_sequence_length, dense_dim), dtype='float32')
    input_mask = Input(shape=(max_sequence_length,), dtype='bool')
    embedding_output = Merge(mode=func, output_shape=lambda x: x[0])([input, input_mask])
    embedding_output = Masking(mask_value=0.0)(embedding_output)
    output = GRU(dense_dim)(embedding_output)
    output_reverse = GRU(dense_dim, go_backwards=True)(embedding_output)
    output = Merge(mode='concat', concat_axis=-1)([output, output_reverse])
    #embedding_input = TimeDistributed(Dense(dense_dim, activation='sigmoid'))(input)
    #output = Merge(mode=func, output_shape=lambda x: x[0])([embedding_input, input_mask])
    #output = MaxPooling1D(pool_length=max_sequence_length)(output)
    #output = Flatten()(output)
    model = Model(input=[input, input_mask], output=output)
    return model

gru_m = gru_model()
gru_sum_embedding_sen_1 = gru_m([embedding_sen_1, sen_1_mask])
gru_sum_embedding_sen_2 = gru_m([embedding_sen_2, sen_2_mask])

abs_merge = lambda x: tf.abs(x[0] - x[1])
mul_merge = lambda x: tf.mul(x[0], x[1])
abs_feature = Merge(mode=abs_merge, output_shape=lambda x: x[0])([sum_embedding_sen_1, sum_embedding_sen_2])
mul_feature = Merge(mode=mul_merge, output_shape=lambda x: x[0])([sum_embedding_sen_1, sum_embedding_sen_2])
gru_abs_feature = Merge(mode=abs_merge, output_shape=lambda x: x[0])([gru_sum_embedding_sen_1, gru_sum_embedding_sen_2])
gru_mul_feature = Merge(mode=mul_merge, output_shape=lambda x: x[0])([gru_sum_embedding_sen_1, gru_sum_embedding_sen_2])



feature = Merge(mode='concat', concat_axis=-1)([abs_feature, mul_feature, gru_abs_feature, gru_mul_feature ])
feature = Dropout(dropout_rate)(feature)
feature = Dense(1, activation='sigmoid')(feature)
final_model = Model(input=[sen_1, sen_1_mask, sen_2, sen_2_mask], output=feature)


optimizer = Nadam(lr=2e-3, clipnorm=5.)
final_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# train data
X_q1 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q1.npy", mmap_mode='r')
X_q2 = np.load("/home/huangzhengjie/quora_pair/quora_stack/stage0_output/train/X_q2.npy", mmap_mode='r')


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
        train_q1_mask = train_q1 > 0
        train_q2_mask = train_q2 > 0
        train_y = y_label[train_index]

        val_q1 = X_q1[test_index][:,:max_sequence_length]
        val_q2 = X_q2[test_index][:,:max_sequence_length]

        val_q1_mask = val_q1 > 0
        val_q2_mask = val_q2 > 0

        val_y = y_label[test_index]
        final_model.fit([train_q1, train_q1_mask, train_q2, train_q2_mask ],
                        train_y,
                        validation_data=([val_q1, val_q1_mask, val_q2, val_q2_mask ], val_y),
                        batch_size=512,
                        nb_epoch=2)
        val_prediction = final_model.predict([val_q1, val_q1_mask, val_q2, val_q2_mask ], batch_size=512, verbose=1)
        score = log_loss(val_y, np.array(val_prediction, "float64"))
        return val_prediction, score

    def train_all(self):
        self.reset()
        train_q1 = X_q1[:,:max_sequence_length]
        train_q2 = X_q2[:,:max_sequence_length]
        train_q1_mask = train_q1 > 0
        train_q2_mask = train_q2 > 0
        train_y = y_label

        final_model.fit([train_q1, train_q1_mask, train_q2, train_q2_mask],
                        train_y,
                        batch_size=512,
                        nb_epoch=2)

        return


    def test(self):
        test_q1 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q1.npy')[:,:max_sequence_length]
        test_q2 = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage0_output/test/X_q2.npy')[:,:max_sequence_length]
        test_q1_mask = test_q1 > 0#, test_q1 == embedding_matrix.shape[0] - 1)
        test_q2_mask = test_q2 > 0#, test_q2 == embedding_matrix.shape[0] - 1)
        prediction = final_model.predict([test_q1, test_q1_mask, test_q2, test_q2_mask], batch_size=512, verbose=1)
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
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/no_leak_deepnet_%s.npy" % state, X_prediction)
    model.train_all()
    pred = model.test()
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/test/no_leak_deepnet_%s.npy" % state, pred)




