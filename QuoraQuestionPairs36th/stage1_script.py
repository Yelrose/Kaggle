import os
import time
from sklearn.model_selection import ParameterGrid

ret = os.system("lkdsfjdsakjf")
param_grid = {"train_dup": [0, 1], "am" : [ 1], "input_left": [0, 1, 2, 3], "input_right": [0, 1, 2, 3]}
input_id = ["", "_clean", "_sc", "_lcs"]
for para in ParameterGrid(param_grid):
    if para['input_left'] < para['input_right']: continue
    para['input_left'] = input_id[para['input_left']]
    para['input_right'] = input_id[para['input_right']]
    ret = os.system("CUDA_VISIBLE_DEVICES=1 python stage1_model/no_leak_random_net.py %s %s \"%s\" \"%s\"" % (para['train_dup'], para['am'], para['input_left'], para['input_right']))
    if ret != 0:
        print 'Error'
        print para
        break


