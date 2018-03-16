Text Generator:

This script builds a simple LSTM model in keras, trains it, and makes predictions.

It generates text by predicting the n+1 character that should follow the nth 
characters. Single characters are treated as classes. The problem can be thought 
of as a classification problem where a sequence of characters is classified;
the number of classes is equal to the number of unique characters in the input.

This is based upon a machinelearningmastery tutorial. 


Usage:

./textgenerator


