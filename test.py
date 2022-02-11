from operator import concat
from tkinter import EXCEPTION
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import unicodedata
import re
import contractions
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer


###Cargando datasets
df = pd.read_csv('./tf_idf_final_data.csv')

print(df.head())