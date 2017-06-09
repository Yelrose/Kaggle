import numpy as np
import pickle as pkl
import sys
import pandas as pd
import json
import xgboost as xgb
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# ensemble different models




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



#std = StandardScaler()
#std.fit(train_leakage_feature)
#train_leakage_feature = std.transform(train_leakage_feature)
#test_leakage_feature= std.transform(test_leakage_feature)




x_train = train_leakage_feature
x_test = test_leakage_feature
# get duplicate flag
y_train = np.load('/home/huangzhengjie/quora_pair/quora_stack/label.npy')





params = {}
params['booster'] = 'gbtree'
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.1
params['nthread'] = 8
params['max_depth'] = 6
params['subsample'] = 0.6
params['min_child_weight'] = 1
params['base_score'] = 0.2
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
    bst = xgb.train(params, d_train, 300, watchlist, verbose_eval=50)
    val_pred = bst.predict(d_valid)
    xgb_train_output[test_index] = val_pred
    score = log_loss(y_train[test_index], np.array(val_pred, dtype='float64'))
    score_log.append(score)
    print 'Fold', fold_id, 'log_loss:', score

state = "%s_%s" % (np.average(score_log), np.std(score_log))
np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/xgb_gbtree_%s.npy" % state, xgb_train_output)
# train all
d_train = xgb.DMatrix(x_train, label=y_train)
watchlist = [(d_train, 'train')]
bst = xgb.train(params, d_train, 300, watchlist, verbose_eval=50)
pred = bst.predict(d_test)
np.save("/home/huangzhengjie/quora_pair/quora_stack/stage1_output/test/xgb_gbtree_%s.npy" % state, pred)
