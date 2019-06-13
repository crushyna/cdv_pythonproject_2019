import collections
import numpy as np
import regex as re
#import operator
from collections import defaultdict

categories = [
  ('5', ['great', 'awesome', 'best', 'original', 'recommend', 'always', 'fabolous', 'love', 'favourite', 'excellent', 'amazing', 'favourite']),
  ('4', ['good', 'nice', 'stylish', 'recommend', 'cute', 'but', 'great', 'specific']),
  ('3', ['average', 'typical', 'mediocre', 'medium', 'but', 'casual', 'specific', 'skip', 'small', 'typical', 'not']),
  ('2', ['bad', 'wrong', 'but',  "don't", 'rude', 'skip', 'small', 'not', 'terrible', 'if']),
  ('1', ['fatal', 'disgusting', 'bad', "don't", 'never', 'worst', 'awful', 'disaster', 'disastrous', 'terrible', 'unaccteptable', 'rude'])
]

def classify2(data):
    # count word frequencies from review
    results_dictionary = {}
    wordfreq = []
    data = data.split()
    for each_review in data:
        wordfreq.append(data.count(each_review))

    # zip-up word count with words from review
    pairs = dict(zip(data, wordfreq))

    # initializing lists for later use
    review_note = []
    ratio_values = []

    for each_key, each_value in categories:

        intersection = list(set(each_value) & set(data))

        # check results here
        ratio = (len(intersection)/len(pairs))
        #print("Calculated ratio for key: ")
        #print(ratio)
        #print("Key: ")
        #print(each_key)
        #print("\n")
            
        ratio_values.append(ratio)
        review_note.append(each_key)

    # zip-up results!        
    results_dictionary = dict(zip(review_note, ratio_values))

    # return result here
    return(int(max(results_dictionary, key=results_dictionary.get)))