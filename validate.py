#!/usr/bin/env python

"Load data, create the validation split, train a random forest, evaluate"
"uncomment the appropriate lines to save processed data to disk"

import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

input_file = 'data/orig/numerai_training_data.csv'

#

d = pd.read_csv( input_file )

# indices for validation examples
iv = d.validation == 1

val = d[iv].copy()
train = d[~iv].copy()

# no need for validation flag anymore
train.drop( 'validation', axis = 1 , inplace = True )

# move the target column to front
cols = train.columns
cols = cols.insert( 0, 'target' )
cols = cols[:-1]

train = train[cols]
val = val[cols]

# train.to_csv( 'data/train_v.csv', index = False )
# val.to_csv( 'data/test_v.csv', index = None )

# encode the categorical variable as one-hot, drop the original column afterwards

train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies.astype( int )), axis = 1 )
# train_num.to_csv( 'data/train_v_num.csv', index = False )

val_dummies = pd.get_dummies( val.c1 )
val_num = pd.concat(( val.drop( 'c1', axis = 1 ), val_dummies.astype(int) ), axis = 1 )
# val_num.to_csv( 'data/test_v_num.csv', index = False )

# train, predict, evaluate

n_trees = 100

rf = RF( n_estimators = n_trees, verbose = True )
rf.fit( train_num.drop( 'target', axis = 1 ), train_num.target )

p = rf.predict_proba( val_num.drop( 'target', axis = 1 ))
p_bin = rf.predict( val_num.drop( 'target', axis = 1 ))

acc = accuracy( val_num.target.values, p_bin )
auc = AUC( val_num.target.values, p[:,1] )
print "AUC: {:.2%}, accuracy: {:.2%}".format( auc, acc )
	
# AUC: 51.40%, accuracy: 51.14%	/ 100 trees
# AUC: 52.16%, accuracy: 51.62%	/ 1000 trees
	