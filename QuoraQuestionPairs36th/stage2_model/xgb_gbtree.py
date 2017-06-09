import numpy as np
import pickle as pkl
import sys
import pandas as pd
import json
import xgboost as xgb
import glob
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# ensemble different models

stage1_model = glob.glob('/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/*.npy')
stage1_model = [wd.split("/")[-1] for wd in stage1_model ]
stage1_model = [wd for wd in stage1_model if wd[:9] == 'hash_deep']
print stage1_model

x_train = []
x_test = []
for mn in stage1_model:
    x_mn_train = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/' + mn)
    x_mn_test = np.load('/home/huangzhengjie/quora_pair/quora_stack/stage1_output/test/' + mn)
    print x_mn_train.shape
    print x_mn_test.shape
    if len(x_mn_train.shape) == 1:
        x_mn_train = np.expand_dims(x_mn_train, -1)
    if len(x_mn_test.shape) == 1:
        x_mn_test = np.expand_dims(x_mn_test, -1)
    x_train.append(x_mn_train)
    x_test.append(x_mn_test)
x_train = np.concatenate(x_train, axis=-1)
x_test = np.concatenate(x_test, axis=-1)


corr = np.corrcoef(x_train.T)
for i, mn1 in enumerate(stage1_model):
    print mn1
    for j , mn2 in enumerate(stage1_model):
        if i == j : continue
        print  '-'*8, '\t',mn2 ,  "%.3lf" % corr[i][j]

# get duplicate flag
y_train = np.load('/home/huangzhengjie/quora_pair/quora_stack/label.npy')





params = {}
params['booster'] = 'gbtree'
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.01
params['nthread'] = 8
params['max_depth'] = 2
params['subsample'] = 0.6
#params['min_child_weight'] = 1
#params['base_score'] = 0.2
params['colsample_bytree'] = 0.8

d_test = xgb.DMatrix(x_test)
y_pred  = None

with open("/home/huangzhengjie/quora_pair/quora_stack/kdf.pkl", 'r') as f:
    kfd = pkl.load(f)

xgb_train_output = np.zeros(len(x_train))
score_log = []
for fold_id, (train_index, test_index) in enumerate(kfd):
    d_train = xgb.DMatrix(x_train[train_index], label=y_train[train_index])#, weight=weight_train[train_index])
    d_valid = xgb.DMatrix(x_train[test_index], label=y_train[test_index])#, weight=weight_train[test_index])
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, 10000, watchlist, verbose_eval=5)
    val_pred = bst.predict(d_valid)
    xgb_train_output[test_index] = val_pred
    score = log_loss(y_train[test_index], np.array(val_pred, dtype='float64'))
    score_log.append(score)
    print 'Fold', fold_id, 'log_loss:', score

state = "%s_%s" % (np.average(score_log), np.std(score_log))
np.save("/home/huangzhengjie/quora_pair/quora_stack/stage2_output/train/xgb_gbtree_%s.npy" % state, xgb_train_output)
# train all
d_train = xgb.DMatrix(x_train, label=y_train)
watchlist = [(d_train, 'train')]
bst = xgb.train(params, d_train, 100, watchlist, verbose_eval=50)
pred = bst.predict(d_test)
np.save("/home/huangzhengjie/quora_pair/quora_stack/stage2_output/test/xgb_gbtree_%s.npy" % state, pred)
