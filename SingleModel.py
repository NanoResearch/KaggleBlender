# Single model

import time
import math
import numpy as np
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

# model
KBEST = 400
MODEL = RandomForestClassifier(max_depth=50, n_estimators=500, max_features=50, criterion='entropy', random_state=1)

# load data
X = np.load("train_data.npy")
y = np.ravel(np.load("train_label.npy"))
X_test = np.load("test_data.npy")

# select KBest
selector = SelectKBest(f_classif, k=KBEST)
X = selector.fit_transform(X, y)
X_test = selector.transform(X_test)

start = time.clock()
		
# fit on train and predict validation and test
MODEL.fit(X, y)
y_test_pred = MODEL.predict_proba(X_test)
		
print "   - " + str(math.floor(time.clock() - start))

test_label = np.load("test_label.npy")
data = np.transpose(np.vstack((np.ravel(test_label), y_test_pred[:,1])))
np.savetxt('results_blender.csv', data, fmt='%d,%f', header='ID,PredictedProb', delimiter=',', comments='')