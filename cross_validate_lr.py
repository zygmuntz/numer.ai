#!/usr/bin/env python

"cross-validation"

import pandas as pd

from sklearn import cross_validation as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

#

def train_and_evaluate( y_train, x_train, y_val, x_val ):

	lr = LR()
	lr.fit( x_train, y_train )

	p = lr.predict_proba( x_val )
	p_bin = lr.predict( x_val )

	acc = accuracy( y_val, p_bin )
	auc = AUC( y_val, p[:,1] )
	
	return ( auc, acc )
	
def transform_train_and_evaluate( transformer ):
	
	global x_train, x_val, y_train
	
	x_train_new = transformer.fit_transform( x_train )
	x_val_new = transformer.transform( x_val )
	
	return train_and_evaluate( y_train, x_train_new, y_val, x_val_new )

#

input_file = 'data/orig/numerai_training_data.csv'

train = pd.read_csv( input_file )
train.drop( 'validation', axis = 1 , inplace = True )

# encode the categorical variable as one-hot, drop the original column afterwards

train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies ), axis = 1 )

# 

"""
cv = CV.KFold( n = len( train ), n_folds = 10 )
aucs = []

for train_i, test_i in cv:

	y_train = train_num.target[train_i].values
	x_train = train_num.drop( 'target', axis = 1 ).loc[train_i]
	
	y_val = train_num.target[test_i].values
	x_val = train_num.drop( 'target', axis = 1 ).loc[test_i]	

	auc, acc = train_and_evaluate( y_train, x_train, y_val, x_val )
	aucs.append( auc )
	
	print "AUC: {:.2%}, accuracy: {:.2%} \n".format( auc, acc )
	
print "avg AUC: {:.2%}".format( sum( aucs ) / len( aucs ))
"""

scores = CV.cross_val_score( LR(), train_num.drop( 'target', axis = 1 ), train_num.target, 
	scoring = 'roc_auc', cv = 10, verbose = 1 )

#

poly_scaled = Pipeline([ ('poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])

transformers = ( MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),
	Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ),
	PolynomialFeatures(), poly_scaled )

for transformer in transformers:

	#x_train_new = transformer.fit_transform( x_train )
	#x_val_new = transformer.transform( x_val )
	
	# fit_transform on the whole set
	train_transformed = transformer.fit_transform( train_num.drop( 'target', axis = 1 ))

	print transformer
	
	scores = CV.cross_val_score( LR(), train_transformed, train_num.target, 
		scoring = 'roc_auc', cv = 10, verbose = 1 )	
	
	print "mean AUC: {:.2%}, std: {:.2%} \n".format( scores.mean(), scores.std())

"""
MaxAbsScaler(copy=True)
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   11.0s finished
mean AUC: 53.42%, std: 0.78%

MinMaxScaler(copy=True, feature_range=(0, 1))
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   10.9s finished
mean AUC: 53.42%, std: 0.78%

RobustScaler(copy=True, with_centering=True, with_scaling=True)
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   10.6s finished
mean AUC: 53.43%, std: 0.78%

StandardScaler(copy=True, with_mean=True, with_std=True)
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:   12.4s finished
mean AUC: 53.43%, std: 0.78%

Normalizer(copy=True, norm='l1')
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    6.4s finished
mean AUC: 52.55%, std: 0.65%

Normalizer(copy=True, norm='l2')
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    7.6s finished
mean AUC: 52.62%, std: 0.61%

Normalizer(copy=True, norm='max')
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    7.2s finished
mean AUC: 52.91%, std: 0.50%

PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  8.4min finished
mean AUC: 53.34%, std: 0.75%

Pipeline(steps=[('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=
False)), ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))])
[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:  5.3min finished
mean AUC: 53.78%, std: 0.76%
"""



"""
original validation split, for comparison

No transformation
AUC: 52.67%, accuracy: 52.74%

MaxAbsScaler(copy=True)
AUC: 53.52%, accuracy: 52.46%

MinMaxScaler(copy=True, feature_range=(0, 1))
AUC: 53.52%, accuracy: 52.48%

RobustScaler(copy=True, with_centering=True, with_scaling=True)
AUC: 53.52%, accuracy: 52.45%

StandardScaler(copy=True, with_mean=True, with_std=True)
AUC: 53.52%, accuracy: 52.42%

Normalizer(copy=True, norm='l1')
AUC: 53.16%, accuracy: 53.19%

Normalizer(copy=True, norm='l2')
AUC: 52.92%, accuracy: 53.20%

Normalizer(copy=True, norm='max')
AUC: 53.02%, accuracy: 52.66%

PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
AUC: 53.25%, accuracy: 52.61%

Pipeline(steps=[
('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), 
('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))])
AUC: 53.62%, accuracy: 53.04%
"""