import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import collections, re
import numpy as np
import xgboost
from src.nlp.text_processor import TextProcessor
from src.classification.bag_of_word_classifier import classify2
from sklearn import model_selection, preprocessing, linear_model, ensemble
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

# klasyfikaca naszym zestawem
our_classification = []
for index, each_review in reviews_and_stars[['text']].itertuples():
  our_classification.append(classify2(each_review.lower()))

reviews_and_stars['our_classification'] = our_classification
reviews_and_stars['guessed_ok'] = np.where(reviews_and_stars['stars'] == reviews_and_stars['our_classification'], 1, 0)

classification_ratio = ((reviews_and_stars['guessed_ok'].sum()) / (len(reviews_and_stars.index)))
regular_classifier_results = {'Classifier': ['Regular Words'], 'Accuracy': classification_ratio}
df_rc = pd.DataFrame(data=regular_classifier_results)
df_rc.set_index('Classifier', inplace=True)

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
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(reviews_and_stars['text_normalized'], reviews_and_stars['stars'], test_size=0.33)
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

# TF-IDF
## word level tf-idf
print("TF-IDF word level transformation...")
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
tfidf_vect.fit(reviews_and_stars['text_normalized'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)
print("Done!")

# dopasowanie modelu
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    print('prediciton sample 0: ', classifier.predict(feature_vector_train[0]))
    print('expected sample 0: ', label[0])
    return metrics.accuracy_score(predictions, valid_y)

# klasyfikator 1: classifier linear na TF-IDF (word level)
print("Linear classification: WordLevel TF-IDF in progress... ")
accuracy1 = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
linear_results = {'Classifier': ['Linear'], 'Accuracy': accuracy1}
df_linear = pd.DataFrame(data=linear_results)
df_linear.set_index('Classifier', inplace=True)

# klasyfikator 2: RF on Word Level TF IDF Vectors
print("Random Forrest, WordLevel TF-IDF classification in progress... ")
accuracy2 = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
randomfor_results = {'Classifier': ['Random Forest'], 'Accuracy': accuracy2 }
df_randomfor = pd.DataFrame(data=randomfor_results)
df_randomfor.set_index('Classifier', inplace=True)

# klasyfikator 3: Extereme Gradient Boosting on Word Level TF IDF Vectors
print("Extereme Gradient Boosting, WordLevel TF-IDF classification in progress...")
accuracy3 = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
xgb_results = {'Classifier': ['Extreme Gradient Boosting'], 'Accuracy': accuracy3}
df_xgb = pd.DataFrame(data=xgb_results)
df_xgb.set_index('Classifier', inplace=True)
classifiers_list = [df_rc, df_linear,  df_randomfor, df_xgb]
classifiers_dataframe = pd.concat(classifiers_list, sort=False)
print(classifiers_dataframe)

# wykres wyników
sns.set(style="whitegrid")
result_diagram = classifiers_dataframe[['Accuracy']].dropna()
catplot = sns.catplot(x='Accuracy', y='Classifier', data=result_diagram.reset_index(), height=8, aspect=2, kind="bar", palette="muted")
plt.title("WordLevel TF-IDF comparision")
plt.show()