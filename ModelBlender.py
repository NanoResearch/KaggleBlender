# Blender
#
# Takes the train/test data/label from FeatureEngineer.py
# Runs the base models on 80% of the data, trains a logistic regression blender on the remaining 20%
# Retrains the base models on 100% of the data and combines the results with the blender

import time
import math
import numpy as np
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# models
KBEST = 400
NAMES = ["RF","LR","ET"]
MODELS = 	[
			RandomForestClassifier(max_depth=10, n_estimators=100, max_features=10, criterion='entropy', random_state=1),
			LogisticRegression(C=15),
			ExtraTreesClassifier(max_depth=10, n_estimators=100, max_features=10, criterion='entropy', random_state=1)
			]
			
def main():
	
	all_start = time.clock()

	# load data
	X = np.load("train_data.npy")
	y = np.ravel(np.load("train_label.npy"))
	X_test = np.load("test_data.npy")
	test_label = np.load("test_label.npy")
	
	# select KBest
	selector = SelectKBest(f_classif, k=KBEST)
	X = selector.fit_transform(X, y)
	X_test = selector.transform(X_test)

	# split data into train and validation
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
	
	### TRAIN MODELS ON 80% OF THE DATA ###
	
	# prediction holders
	VAL_LIST = []
	
	for MODEL in MODELS:
		
		print MODEL
		
		start = time.clock()
		
		# fit on train and predict validation and test
		MODEL.fit(X_train, y_train)
		y_val_pred = MODEL.predict_proba(X_val)
		
		print " - " + str(log_loss(y_val, y_val_pred, eps=1e-15, normalize=True))
		
		# save validation
		VAL_LIST.append(y_val_pred[:,1])
		
		print "   - " + str(math.floor(time.clock() - start))
		
	# combine models
	val_data = np.column_stack(tuple(VAL_LIST))
	
	#######################################
	
	# create and fit blending model
	model = LogisticRegression(C=15)
	model.fit(val_data, y_val)
	
	### RETRAIN MODELS ON 100% OF THE DATA ###
	
	# prediction holders
	TEST_LIST = []
	
	for NAME,MODEL in zip(NAMES,MODELS):
		
		print MODEL
		
		start = time.clock()
		
		# fit on train and predict validation and test
		MODEL.fit(X, y)
		y_test_pred = MODEL.predict_proba(X_test)
				
		# save
		TEST_LIST.append(y_test_pred[:,1])
		
		# save result to file
		data = np.transpose(np.vstack((np.ravel(test_label), y_test_pred[:,1])))
		np.savetxt(NAME+'.csv', data, fmt='%d,%f', header='ID,PredictedProb', delimiter=',', comments='')
	
		print "   - " + str(math.floor(time.clock() - start))
		
	# combine models
	test_data = np.column_stack(tuple(TEST_LIST))
	
	##########################################
	
	# predict and save result
	predicted_label = model.predict_proba(test_data)
	data = np.transpose(np.vstack((np.ravel(test_label), predicted_label[:,1])))
	np.savetxt('results_blender.csv', data, fmt='%d,%f', header='ID,PredictedProb', delimiter=',', comments='')
	
	# print
	print model.intercept_
	print model.coef_
	print time.clock() - all_start
	
main()