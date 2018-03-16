#!/usr/bin/env python
#
#
#
# Robert Mitchell
#

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

import numpy as np
import string
import os
import sys


#===============================================================================
#
#                          Prepare Input for LSTM
#
# read input text, make all lower case, remove strange characters
fileName = 'at_the_mountains_of_madness.txt'
rawText = open(fileName).read()

# convert to lowercase
lowerCaseText = str(rawText.lower())

# strip any digits or punctuation characters that might be present in training text
cleanText = str(lowerCaseText.translate(str.maketrans( string.punctuation + string.digits, ' '*(len(string.punctuation) + len(string.digits)))))

# the network requires numerical input, so convert the text's chars to integers,
# 0 to 26 mapps the alphabet, but there are some punctuation and numbers in
# there as well, len(dict) > 26
chars = sorted(list(set(cleanText)))
charToInt = dict((c, i) for i,c in enumerate(chars))
intToChar = dict((i, c) for i,c in enumerate(chars))

# get total number of characters that make up the text
numberOfCharsInText = len(cleanText)

# count the number of unique characters, this will be our vocabulary
numberOfCharsInVocabulary = len(chars)

# we will define 100 characters to be a pattern.
patternLength = 100

# truncate text so that it's evenly divisible by our patternLength
cleanText = cleanText[0:patternLength*int(len(cleanText)/patternLength)-len(cleanText)]

# break the text into patterns stored as integers and as actual characters.
patternInputs = list() # a pattern is 100 characters from the input text.
patternOutputs = list() # the output for a pattern is 101th character. The output for a pattern is always the very next character.

# build the patterns
for charIndex in range(0, len(cleanText)-patternLength):
    patternInputs.append([charToInt[char] for char in cleanText[charIndex:charIndex + patternLength]])
    patternOutputs.append(charToInt[cleanText[charIndex+patternLength]])

print('Number of training patterns:' + str(len(patternInputs)))

# reshape training data for Keras to be a list as, [samples, timesteps, features]
inputs = np.reshape(patternInputs, (len(patternInputs), patternLength, 1))

# normalize
inputs = inputs/float(numberOfCharsInVocabulary)

# one hot encode our output variable
outputs = np_utils.to_categorical(patternOutputs)

# now our training data is complete, we have inputs and expected outputs.
print('input read')
#===============================================================================
#
#                           Build or Read LSMT From file
#
model = None

if not os.path.isfile('at_the_mountains_of_madness.h5'):
    # we'll use a simple network topology
    model = Sequential()
    model.add(LSTM(256, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(outputs.shape[1], activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
    # train the network
    model.fit(inputs, outputs, epochs = 20, batch_size = 128)
    model.save('text_generator.h5')
else:
    model = load_model('at_the_mountains_of_madness.h5')

#===============================================================================
#
#                                        Predict Text
#
randomIndex = np.random.randint(0, len(inputs)-1)
pattern = list(inputs[randomIndex])

generatedText = str()

for i in range(1000):
    x = np.reshape(pattern, (1,len(pattern),1))
    x = x/float(numberOfCharsInVocabulary)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = intToChar[index]
    #seq_in = [intToChar[value] for value in pattern]
    #sys.stdout.write(result)
    generatedText += result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# print output
print(generatedText)
#===============================================================================
