import collections
import numpy as np
import regex as re
from collections import defaultdict

categories = [
  ('cat5', ['great', 'awesome', 'best', 'original', 'recommend', 'always', 'fabolous', 'love', 'favourite']),
  ('cat4', ['good', 'nice', 'stylish', 'recommend', 'cute']),
  ('cat3', ['average', 'typical', 'mediocre', 'medium', 'but']),
  ('cat2', ['bad', 'wrong', 'bca', 'but']),
  ('cat1', ['fatal', 'disgusting', 'bad', "don't", 'never'])
]

'''
for each_key, each_value in categories:
    print(each_key)
    print(each_value)
'''

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
        print("Calculated ratio for key: ")
        print(ratio)
        print("Key: ")
        print(each_key)
        print("\n")
            
        ratio_values.append(ratio)
        review_note.append(each_key)

    # zip-up results!        
    results_dictionary = dict(zip(review_note, ratio_values))
    
    # return result here
    print(max(results_dictionary.keys()))


def five_stars_reviews_list(data):
    #print(data['text'])
    #dct = defaultdict(list)
    five_text_array = np.asarray(data['text'])
    five_bagsofwords = [collections.Counter(re.findall(r'\w+', txt)) for txt in five_text_array]
    five_sumbags = sum(five_bagsofwords, collections.Counter())
    #highestCount = max(five_sumbags.values())
    #return(five_sumbags.info())
    return(five_sumbags.keys())