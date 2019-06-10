import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections, re
import numpy as np
from src.nlp.text_processor import TextProcessor
#from src.classification.bag_of_word_classifier import classify
from src.classification.bag_of_word_classifier import classify2

# plik źródłowy do analizy
filename = "yelp.csv"
test_review = "very average restaurant in this town. good food, but rather mediocre service."

# przetworzenie danych wejściowych (csv -> dataframe)
oryg_dataframe = pd.read_csv(filename).dropna()

# wstępny podgląd danych
#print(oryg_dataframe.info())
#print(oryg_dataframe.head())
#print(oryg_dataframe[['stars', 'text']].head())
reviews_and_stars = oryg_dataframe[['text', 'stars']]

# klasyfikacja tekstu
five_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 5)]
four_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 4)]
three_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 3)]
two_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 2)]
one_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 1)]

# bag-of-words dla ocen 5
#print(five_stars_reviews[['text']])
classify2(test_review)

#for each_review in five_stars_reviews[['text']]:
  #print(five_stars_reviews[each_review])
 # classify(five_stars_reviews[each_review])

'''
#print(five_stars_reviews['text'])
#five_text_array = np.asarray(five_stars_reviews['text'])
#five_bagsofwords = [collections.Counter(re.findall(r'\w+', txt)) for txt in five_text_array]
#five_sumbags = sum(five_bagsofwords, collections.Counter())
#print(five_sumbags.info())

#print(reviews_and_stars.groupby('stars').describe())

reviews_and_stars['text'] = reviews_and_stars['text'].str.lower()
print(reviews_and_stars)

### creating the feature matrix 
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
X = matrix.fit_transform(reviews_and_stars['text']).toarray()
y = reviews_and_stars['stars']

# split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print(cm)
print(cr)
print(accuracy)
'''
'''
oryg_dataframe['text length'] = oryg_dataframe['Reviews'].apply(len)

sns.set_style('white')
g = sns.FacetGrid(oryg_dataframe ,col='Rating')
g.map(plt.hist,'text length')
plt.show()
'''

# normalizacja

'''
with open(filename, 'r') as myfile:
  data = myfile.read()
  
result = TextProcessor().normalize(data)
output_file = open(filename, "w")
output_file.write(result.replace("__label__2", "\n__label__2" ))
output_file.close()
'''

# wektoryzacja tekstu (TF-IDF)
# + przetworzenie, 3 klasyfikatory

# parsowanie