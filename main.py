# quick_start ze zadanek nlp-tools
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.nlp.text_processor import TextProcessor

# plik źródłowy do analizy
filename = "TA_restaurants_curated.csv"

# przetworzenie danych wejściowych (csv -> dataframe)
# wstępny podgląd danych
oryg_dataframe = pd.read_csv(filename, index_col=0).dropna()
#print(oryg_dataframe.info())
#print(oryg_dataframe.head())
#print(oryg_dataframe[['Number of Reviews', 'Reviews', 'Rating']].head())

# klasyfikacja tekstu
oryg_dataframe['text length'] = oryg_dataframe['Reviews'].apply(len)

sns.set_style('white')
g = sns.FacetGrid(oryg_dataframe ,col='Rating')
g.map(plt.hist,'text length')
plt.show()


# normalizacja

'''with open(filename, 'r') as myfile:
  data = myfile.read()
  
result = TextProcessor().normalize(data)
output_file = open(filename, "w")
output_file.write(result.replace("__label__2", "\n__label__2" ))
output_file.close()
'''

# wektoryzacja tekstu (TF-IDF)
# + przetworzenie, 3 klasyfikatory

# parsowanie