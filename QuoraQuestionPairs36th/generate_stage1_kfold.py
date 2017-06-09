import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle as pkl
k_fold = StratifiedKFold(n_splits=8, shuffle=True)
train_data = pd.read_csv('../train.csv', encoding='utf-8')
label = np.array(train_data[u'is_duplicate'])
kfd = list(k_fold.split(label, label))
with open("kdf.pkl", 'w') as f:
    pkl.dump(kfd, f)

np.save('label.npy', label)
