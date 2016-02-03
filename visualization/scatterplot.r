library( ggplot2 )

input_file = 'numerai_training_data.csv'
d = read.csv( input_file )

f1f2 = ggplot( d, aes( x = f1, y = f2 ))
f1f2 + geom_point()
