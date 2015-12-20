#!/usr/bin/env python

"Load data, train a random forest, output predictions"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF

train_file = 'data/orig/numerai_training_data.csv'
test_file = 'data/orig/numerai_tournament_data.csv'
output_file = 'data/predictions.csv'

#

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# no need for validation flag
train.drop( 'validation', axis = 1 , inplace = True )

# encode the categorical variable as one-hot, drop the original column afterwards
# but first let's make sure the values are the same in train and test

assert( set( train.c1.unique()) == set( test.c1.unique()))

train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies.astype( int )), axis = 1 )

test_dummies = pd.get_dummies( test.c1 )
test_num = pd.concat(( test.drop( 'c1', axis = 1 ), test_dummies.astype(int) ), axis = 1 )

# train and predict

n_trees = 1000

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit( train_num.drop( 'target', axis = 1 ), train_num.target )

p = rf.predict_proba( test_num.drop( 't_id', axis = 1 ))

# save

test_num['probability'] = p[:,1]
test_num.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )



	