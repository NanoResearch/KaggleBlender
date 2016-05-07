# Feature Engineer
#
# Takes as input train.csv and test.csv
# Outputs test_data.csv, train_data.csv, test_label.csv and train_label.csv
#
# Creates 455 features from the 131 original features by taking:
# 112  features: numerical features
# 112  features: log(numerical features)
# 112  features: NaN or not (numerical features)
# 19   features: factorized(categorical features)
# 100  features: one-hot and rbm (categorical features) after splitting the complex categorical features

import math
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.neural_network import BernoulliRBM

other_indices 		= ['v22', 'v56', 'v113', 'v125']
categorical_indices = ['v3', 'v24', 'v30', 'v31', 'v47', 'v52', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91', 'v107', 'v110', 'v112']
numerical_indices 	= [i for i in ['v'+str(j) for j in range(1,132)] if i not in categorical_indices and i not in other_indices]

def main():
	
	# load train and test
	(train_data, train_label) = load(True)
	(test_data, test_label) = load(False)
	
	# add train and test together so we perform the same operations on both
	data, split_at = adding(train_data, test_data)
	
	# combine data horizontally and convert to numpy
	data = combine(data)
	
	# split on train and test
	train_data = data[:split_at,:]
	test_data  = data[split_at:,:]
	
	# save the data as name_train.csv and name_test.csv
	np.save('train_data', train_data)
	np.save('test_data', test_data)
	np.save('train_label', train_label)
	np.save('test_label', test_label)
	
def load(train = True):

	if train:
		filename = "train.csv"
	else:
		filename = "test.csv"
	
	# load data
	all_data = pd.read_csv(filename, delimiter=",")
	
	# get nan values
	nan_data = pd.isnull(all_data)
	
	# get other features
	other = all_data[other_indices]
	
	# get categorical features
	categorical = all_data[categorical_indices]
	
	# get numerical features
	numerical = all_data[numerical_indices]
	
	# get numerical NaNs
	nan = nan_data[numerical_indices]
	
	if train:
		label = all_data[['target']]
	else:
		label = all_data[['ID']]
		
	return (numerical, categorical, other, nan), label
	
def adding(train, test):

	# unpack tuples
	(train_numerical, train_categorical, train_other, train_nan) = train
	(test_numerical,  test_categorical,  test_other,  test_nan)  = test
	
	# get number of rows in train as split index
	(split_at, _) = train_numerical.shape
	
	# add train and test together
	numerical = train_numerical.append(test_numerical)
	categorical = train_categorical.append(test_categorical)
	other = train_other.append(test_other)
	nan = train_nan.append(test_nan)
	
	# pack and return
	data = (numerical, categorical, other, nan)
	return data, split_at
	
def combine(data):
	
	# unpack data
	(numerical, categorical, other, nan) = data
	
	# create numlog (add a little bit to prevent values <= 0)
	numlog = np.log(numerical+0.01)
	numlog = (numlog - numlog.mean()) / (numlog.max() - numlog.min())
	numlog = numlog.fillna(0)
	
	# normalize and impute numerical
	numerical = (numerical - numerical.mean()) / (numerical.max() - numerical.min())
	numerical = numerical.fillna(0)
	
	# RBM categorical
	rbmcat = pd.get_dummies(categorical)
	
	# RBM other
	rbmother = pd.get_dummies(pd.DataFrame(splitcomplex(np.array(other))))
	
	# factorize categorical
	for column in categorical:
		categorical[column],_ = pd.factorize(categorical[column])
	categorical = (categorical - categorical.mean()) / (categorical.max() - categorical.min())
	
	# factorize other
	for column in other:
		other[column],_ = pd.factorize(other[column])
	other = (other - other.mean()) / (other.max() - other.min())
	
	### CONVERT TO NUMPY ###
	numerical = np.array(numerical)
	numlog = np.array(numlog)
	categorical = np.array(categorical)
	rbmcat = np.array(rbmcat)
	other = np.array(other)
	rbmother = np.array(rbmother)
	nan = np.array(nan)
	########################

	# rbm over rbmcat and rbmother
	rbm = BernoulliRBM(n_components=100, batch_size=100, n_iter=50, learning_rate=0.02, verbose=1, random_state=1)
	rbmdata = rbm.fit_transform(np.concatenate((rbmcat, rbmother), axis=1))
	rbmdata = (rbmdata - rbmdata.mean()) / (rbmdata.max() - rbmdata.min())
	
	# normalize nan
	nan = (nan - nan.mean()) / (nan.max() - nan.min())
	
	# concat and return
	data = np.concatenate((numerical, numlog, categorical, other, rbmdata, nan), axis=1)
	return data

def splitcomplex(data):

	columns = []
	
	for i in range(0,4):
		
		# change all items to the same length by padding spaces in front and changing nan to spaces
		maxletters = np.amax([1 if pd.isnull(x) else len(x) for x in data[:,i]])
		
		list = []
		for item in data[:,i]:
		
			if pd.isnull(item):
				list.append(' ' * maxletters)
			else:
				list.append((maxletters - len(item)) * ' ' + item)
			
		# separate every item into maxletters items
		for j in range(0, maxletters):
			
			col = np.asarray([x[j] for x in list])
			columns.append(col)
	
	return np.transpose(np.vstack(columns))
	
	
main()