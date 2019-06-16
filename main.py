import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections, re
import numpy as np
from src.nlp.text_processor import TextProcessor
from src.classification.bag_of_word_classifier import classify2
from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from timeit import default_timer as timer
from sklearn import metrics
import nltk

nltk.download('stopwords')

# plik źródłowy do analizy
filename = "yelp.csv"
test_good_review = "@@@Very average restaurant in this town. Good food... 5 ...but rather mediocre service###."
test_bad_review = "never come back here, disgusting, fatal service"

# przetworzenie danych wejściowych (csv -> dataframe)
print("Importing data...")
oryg_dataframe = pd.read_csv(filename).dropna()

# wstępny podgląd danych
#print(oryg_dataframe.info())
#print(oryg_dataframe.head())
print("Creating new dataframe...")
reviews_and_stars = oryg_dataframe[['text', 'stars']]

# zestawy recenzji + oceny do klasyfikacji
'''
five_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 5)]
four_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 4)]
three_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 3)]
two_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 2)]
one_stars_reviews = reviews_and_stars.loc[(reviews_and_stars['stars'] == 1)]
'''

#testowy bag-of-words dla ocen = 3
#print(five_stars_reviews[['text']])
#print(classify2(test_bad_review))


# klasyfikaca naszym zestawem
'''
our_classification = []
for index, each_review in reviews_and_stars[['text']].itertuples():
  our_classification.append(classify2(each_review.lower()))

reviews_and_stars['our_classification'] = our_classification
reviews_and_stars['guessed_ok'] = np.where(reviews_and_stars['stars'] == reviews_and_stars['our_classification'], 1, 0)

classification_ratio = ((reviews_and_stars['guessed_ok'].sum()) / (len(reviews_and_stars.index))) * 100
print("Precision ratio: ")
print(classification_ratio)
'''

# procesowanie tektsu, lematyzacja, normalizacja
print("Text normalization and lemmatization...")
normalized_reviews = []
for index, each_review in reviews_and_stars[['text']].itertuples():
  normalized_reviews.append(TextProcessor().normalize(each_review))

reviews_and_stars['text_normalized'] = normalized_reviews
print("Done!")
print(reviews_and_stars)

# podział datasetu na treningowy i sprawdzający (training and validation)
print("Creating validation and training dataset...")
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(reviews_and_stars['text_normalized'], reviews_and_stars['stars'])
print("Done!")

# dekodowanie treści
print("Data preprocessing...")
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
print("Done!")

# wektoryzacja (count vectorizer object)
print("Data vectorization...")
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(reviews_and_stars['text_normalized'])
print("Done!")

# transformacja danych treningowych i sprawdzających używając wektoryzacji
print("Training data transformation...")
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
print("Done!")
#print(xtrain_count)

# TF-IDF
## word level tf-idf
print("TF-IDF word level transformation...")
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(reviews_and_stars['text_normalized'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
print("Done!")

'''
## ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(reviews_and_stars['text_normalized'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

## characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(reviews_and_stars['text_normalized'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
'''

# dopasowanie modelu
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

# classifier linear - klasyfikator 1
print("Linear classification: word level (TF-IDF) in progress... ")
accuracy2 = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy2)

## Linear Classifier on Word Level TF IDF Vectors


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

# wektoryzacja tekstu (TF-IDF)
# + przetworzenie, 3 klasyfikatory

# parsowanie