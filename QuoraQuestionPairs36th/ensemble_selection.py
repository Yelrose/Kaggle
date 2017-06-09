import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import sys
import pandas as pd
import glob
import pickle
import os
stage2_output = glob.glob("hyperopt_output/train/*.npy")
stage2_output = [ wd.split("/")[-1] for wd in stage2_output ]
stage2_output =  [  wd for wd in stage2_output if os.path.exists("hyperopt_output/test/" + wd)]

y_train = np.load("/home/huangzhengjie/quora_pair/quora_stack/label.npy", mmap_mode="r")


x_train = []
for mn in stage2_output:
    x_mn_train = np.load('/home/huangzhengjie/quora_pair/quora_stack/hyperopt_output/train/' + mn)
    if len(x_mn_train.shape) == 1:
        x_mn_train = np.expand_dims(x_mn_train, -1)
    x_train.append(x_mn_train)
x_train = np.concatenate(x_train, axis=-1)
x_train = np.array(x_train, dtype='float64')
print x_train.shape

try_turn = 10
while try_turn:
    print 'Turn', 10 - try_turn
    try_turn -= 1
    # First Round
    # Pick K best
    picked_col_set = []
    picked_topk = 5
    bagging_fraction = 0.5
    turn_id = [ i for i in range(x_train.shape[1]) if np.random.uniform() < bagging_fraction]
    print 'Num Samples', len(turn_id)
    top_k = [(i, log_loss(y_train, x_train[:, i])) for i in turn_id]
    print 'ALL Data'
    for k, v in top_k:
        print stage2_output[k]
        print v
    top_k = sorted(top_k, cmp=lambda x, y: cmp(x[1], y[1]))[:picked_topk]
    print 'Topk:'
    print top_k
    # bagging top k

    picked_pred = np.zeros(len(x_train))
    for id, _ in top_k:
        picked_col_set.append(id)
        picked_pred += x_train[:, id]
    picked_loss = log_loss(y_train, picked_pred / len(picked_col_set))
    # starting selection
    print 'Start',  picked_loss
    while True:
        cand_set = [ i for i in turn_id]
        if len(cand_set) == 0:
            print 'no interset'
            break
        cand_set = [ (i, log_loss(y_train, (x_train[:, i] + picked_pred) / (len(picked_col_set) + 1))) for i in cand_set ]
        cand_set = [(k, v) for k, v in cand_set if v < picked_loss - 1e-6]
        if len(cand_set) == 0:
            print 'no improve'
            break
        cand_set = sorted(cand_set, cmp=lambda x, y: cmp(x[1], y[1]))

        picked_id = cand_set[0]
        picked_pred += x_train[:, picked_id[0]]
        picked_loss = picked_id[1]
        picked_col_set.append(picked_id[0])
        print picked_id[0], picked_id[1]

    final_pred = None
    for mn in picked_col_set:
        x_mn_test = np.load('/home/huangzhengjie/quora_pair/quora_stack/hyperopt_output/test/' + stage2_output[mn])
        if np.max(x_mn_test) > 1 or np.min(x_mn_test) < 0:
            print stage2_output[mn]
            print mn, np.max(x_mn_test), np.min(x_mn_test)
            raise
        if len(x_mn_test.shape) == 1:
            x_mn_test = np.expand_dims(x_mn_test, -1)
        if final_pred is None:
            final_pred = np.zeros((len(x_mn_test), 1), dtype='float64')
        final_pred += x_mn_test
    final_pred /= len(picked_col_set)
    for key in picked_col_set:
        print stage2_output[key]
    np.save("submission/ensemble_selection_%s_%s.npy" % (picked_loss, len(picked_col_set)), final_pred)








