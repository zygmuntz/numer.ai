#!/usr/bin/env python

"Load data, create the validation split, optionally scale data, train a linear model, evaluate"
"Code updated for march 2016 data"

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy, log_loss

#

def train_and_evaluate( y_train, x_train, y_val, x_val ):

	lr = LR()
	lr.fit( x_train, y_train )

	p = lr.predict_proba( x_val )

	auc = AUC( y_val, p[:,1] )
	ll = log_loss( y_val, p[:,1] )
	
	return ( auc, ll )
	
def transform_train_and_evaluate( transformer ):
	
	global x_train, x_val, y_train
	
	x_train_new = transformer.fit_transform( x_train )
	x_val_new = transformer.transform( x_val )
	
	return train_and_evaluate( y_train, x_train_new, y_val, x_val_new )
	
#

input_file = 'data/numerai_training_data.csv'

d = pd.read_csv( input_file )
train, val = train_test_split( d, test_size = 5000 )

y_train = train.target.values
y_val = val.target.values

x_train = train.drop( 'target', axis = 1 )
x_val = val.drop( 'target', axis = 1 )

# train, predict, evaluate

auc, ll = train_and_evaluate( y_train, x_train, y_val, x_val )

print "No transformation"
print "AUC: {:.2%}, log loss: {:.2%} \n".format( auc, ll )

# try different transformations for X
# X is already scaled to (0,1) so these won't make much difference

transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),  
	Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ) ]

#poly_scaled = Pipeline([ ( 'poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])
#transformers.append( PolynomialFeatures(), poly_scaled )

for transformer in transformers:

	print transformer
	auc, ll = transform_train_and_evaluate( transformer )
	print "AUC: {:.2%}, log loss: {:.2%} \n".format( auc, ll )

"""
No transformation
AUC: 52.35%, log loss: 69.20%

MaxAbsScaler(copy=True)
AUC: 52.35%, log loss: 69.20%

MinMaxScaler(copy=True, feature_range=(0, 1))
AUC: 52.35%, log loss: 69.20%

RobustScaler(copy=True, with_centering=True, with_scaling=True)
AUC: 52.35%, log loss: 69.20%

StandardScaler(copy=True, with_mean=True, with_std=True)
AUC: 52.35%, log loss: 69.20%

Normalizer(copy=True, norm='l1')
AUC: 51.26%, log loss: 69.26%

Normalizer(copy=True, norm='l2')
AUC: 52.18%, log loss: 69.21%

Normalizer(copy=True, norm='max')
AUC: 52.40%, log loss: 69.19%
"""