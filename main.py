# quick_start ze zadanek nlp-tools
import json
import pandas as pd
from src.nlp.text_processor import TextProcessor

# plik źródłowy do analizy
filename = "TA_restaurants_curated.csv"

# przetworzenie danych wejściowych (csv -> dataframe)
oryg_dataframe = pd.read_csv(filename)
print(oryg_dataframe.info())
print(oryg_dataframe.head())

# klasyfikacja tekstu


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