import numpy as np
import pickle as pkl
import sys
import pandas as pd
import glob
import json
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import ensemble
# ensemble different models


stage1_model = glob.glob('/home/huangzhengjie/quora_pair/quora_stack/stage1_output/train/*.npy')
stage1_model = [wd.split("/")[-1] for wd in stage1_model]
stage1_model = [wd for wd in stage1_model if wd[:7] == 'no_leak']
print stage1_model



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




sys.stdout.flush()
std = StandardScaler()
std.fit(np.vstack([train_leakage_feature, test_leakage_feature]))
train_leakage_feature = std.transform(train_leakage_feature)
test_leakage_feature = std.transform(test_leakage_feature)


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



x_train = train_leakage_feature
x_test = test_leakage_feature
# get duplicate flag
y_train = np.load('/home/huangzhengjie/quora_pair/quora_stack/label.npy')





with open("/home/huangzhengjie/quora_pair/quora_stack/kdf.pkl", 'r') as f:
    kfd = pkl.load(f)

base_model = [
        # linear model
        #('linear_regression', linear_model.LinearRegression(), False),
        ('logistic_regression', linear_model.LogisticRegression(C=0.7), True),
        #('ridge', linear_model.Ridge(), False),
        # Tree model
        #('random_forest_classifier', ensemble.RandomForestClassifier(n_estimators=100, verbose=True, max_depth=6), True),
        #('random_forest_regression', ensemble.RandomForestRegressor(n_estimators=100, verbose=True, max_depth=6), False),
        # KNN
        #("knn_3", KNeighborsRegressor(n_neighbors=3), False),
        #("knn_5", KNeighborsRegressor(n_neighbors=5), False),
        #("knn_8", KNeighborsRegressor(n_neighbors=8), False),
        ]
clf_id = 0
for clf_name, clf, p_flag in base_model:
    print clf_name, 'Start'
    xgb_train_output = np.zeros((len(x_train), ))
    test_pred = np.zeros((len(x_test), ))
    score_log = []
    for fold_id, (train_index, test_index) in enumerate(kfd):
        clf.fit(x_train[train_index], y_train[train_index])
        if p_flag:
            val_pred = clf.predict_proba(x_train[test_index])[:, 1]
        else:
            val_pred = clf.predict(x_train[test_index])
        xgb_train_output[test_index] = val_pred
        score = log_loss(y_train[test_index], np.array(val_pred, dtype='float64'))
        score_log.append(score)
        print clf_name, 'Fold', fold_id, 'log_loss:', score
    clf.fit(x_train, y_train)
    if p_flag:
        pred = clf.predict_proba(x_test)[:, 1]
    else:
        pred = clf.predict(x_test)
    test_pred[:] = pred
    state = "%s_%s" % (np.average(score_log), np.std(score_log))
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage2_output/train/%s_%s.npy" %(clf_name, state), xgb_train_output)
    np.save("/home/huangzhengjie/quora_pair/quora_stack/stage2_output/test/%s_%s.npy" % (clf_name, state), test_pred)
    clf_id += 1
