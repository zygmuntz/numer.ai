#!/usr/bin/env python

"check validation consistency of predictions"

import sys
import pandas as pd

from math import log
from sklearn.metrics import log_loss

try:
	submission_file = sys.argv[1]
except IndexError:	
	submission_file = 'predictions.csv'
	
try:
	test_file = sys.argv[2]
except IndexError:	
	test_file = 'data/test.csv'	

try:
	print( "loading {}...".format( submission_file ))
	s = pd.read_csv( submission_file )
except:
	print( "\nUsage: check_consistency.py <predictions file> <test file>" )
	print( "  i.e. check_consistency.py p.csv numerai_tournament_data.csv\n" )
	raise SystemExit

print( "loading {}...\n".format( test_file ))
test = pd.read_csv( test_file )

v = test[ test.data_type == 'validation' ].copy()
v = v.merge( s, on = 'id', how = 'left' )

eras = v.era.unique()

good_eras = 0

for era in eras:
	tmp = v[ v.era == era ]
	ll = log_loss( tmp.target, tmp.probability )
	is_good = ll < -log( 0.5 )
	
	if is_good:
		good_eras += 1
	
	print( "{} {} {:.2%} {}".format( era, len( tmp ), ll, is_good ))
	
consistency = good_eras / float( len( eras ))
print( "\nconsistency: {:.1%} ({}/{})".format( consistency, good_eras, len( eras )))

ll = log_loss( v.target, v.probability )
print( "log loss:    {:.2%}\n".format( ll ))
