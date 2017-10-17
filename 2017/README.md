# Numerai 2017

Code for checking consistency of predictions. Usage:

`python check_consistency.py <predictions file> <test file>`

Example output:

	$ ./check_consistency.py predictions/p.csv data/test.csv
	loading predictions/p.csv...
	loading data/test.csv...
	
	era86 6091 69.06% True
	era87 6079 68.61% True
	era88 6067 69.28% True
	era89 6064 69.12% True
	era90 6050 68.90% True
	era91 6027 69.22% True
	era92 6048 69.85% False
	era93 6038 69.34% False
	era94 6326 68.94% True
	era95 6349 68.88% True
	era96 6336 69.23% True
	era97 6390 69.31% True

	consistency: 83.3% (10/12)
	log loss:    69.14%
	
	