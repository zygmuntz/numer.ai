#!/usr/bin/env python

"Load data, scale, train a linear model, output predictions"

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression as LR

train_file = 'data/orig/numerai_training_data.csv'
test_file = 'data/orig/numerai_tournament_data.csv'
output_file = 'data/predictions_lr.csv'

#

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# no need for validation flag
train.drop( 'validation', axis = 1 , inplace = True )

# encode the categorical variable as one-hot, drop the original column afterwards
# but first let's make sure the values are the same in train and test

assert( set( train.c1.unique()) == set( test.c1.unique()))

train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies ), axis = 1 )

test_dummies = pd.get_dummies( test.c1 )
test_num = pd.concat(( test.drop( 'c1', axis = 1 ), test_dummies ), axis = 1 )

#

y_train = train_num.target.values

x_train = train_num.drop( 'target', axis = 1 )
x_test = test_num.drop( 't_id', axis = 1 )

print "transforming..."

pipeline = MinMaxScaler()
#pipeline = Pipeline([ ('poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])

x_train_new = pipeline.fit_transform( x_train )
x_test_new = pipeline.transform( x_test )

print "training..."

lr = LR()
lr.fit( x_train_new, y_train )

print "predicting..."

p = lr.predict_proba( x_test_new )

print "saving..."

test_num['probability'] = p[:,1]
test_num.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )

# AUC 0.51706 / no transformations
# AUC 0.52781 / min-max scaler
# AUC 0.51784 / poly features + min-max scaler
	