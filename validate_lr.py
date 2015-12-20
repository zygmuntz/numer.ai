#!/usr/bin/env python

"Load data, create the validation split, optionally scale data, train a linear model, evaluate"

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
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

d = pd.read_csv( input_file )

# indices for validation examples
iv = d.validation == 1

val = d[iv].copy()
train = d[~iv].copy()

# no need for validation flag anymore
train.drop( 'validation', axis = 1 , inplace = True )
val.drop( 'validation', axis = 1 , inplace = True )

# encode the categorical variable as one-hot, drop the original column afterwards

train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies ), axis = 1 )

val_dummies = pd.get_dummies( val.c1 )
val_num = pd.concat(( val.drop( 'c1', axis = 1 ), val_dummies ), axis = 1 )

# 

y_train = train_num.target.values
y_val = val_num.target.values

x_train = train_num.drop( 'target', axis = 1 )
x_val = val_num.drop( 'target', axis = 1 )

# train, predict, evaluate

auc, acc = train_and_evaluate( y_train, x_train, y_val, x_val )

print "No transformation"
print "AUC: {:.2%}, accuracy: {:.2%} \n".format( auc, acc )

# try different transformations for X

transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),  
	Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ),
	PolynomialFeatures() ]

poly_scaled = Pipeline([ ( 'poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])
transformers.append( poly_scaled )

for transformer in transformers:

	print transformer
	auc, acc = transform_train_and_evaluate( transformer )
	print "AUC: {:.2%}, accuracy: {:.2%} \n".format( auc, acc )

"""
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