#!/usr/bin/env python

"Load data, scale, train a linear model, output predictions"

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression as LR

train_file = 'data/numerai_training_data.csv'
test_file = 'data/numerai_tournament_data.csv'
output_file = 'data/predictions_lr.csv'

#

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

#

y_train = train.target.values

x_train = train.drop( 'target', axis = 1 )
x_test = test.drop( 't_id', axis = 1 )

print "training..."

lr = LR()
lr.fit( x_train, y_train )

print "predicting..."

p = lr.predict_proba( x_test )

print "saving..."

test['probability'] = p[:,1]
test.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )

# 0.69101 public
	