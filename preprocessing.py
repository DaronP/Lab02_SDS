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
enronSpamSubset = pd.read_csv('./enronSpamSubset.csv')
completeSpamAssassin = pd.read_csv('./completeSpamAssassin.csv')

###Visualizando la data
#print(enronSpamSubset.head())
#print(completeSpamAssassin.head())

#print(len(enronSpamSubset.columns), enronSpamSubset.columns)
#print(len(completeSpamAssassin.columns), completeSpamAssassin.columns)


enronSpamSubset = enronSpamSubset.drop('Unnamed: 0.1', axis=1)

#print(len(enronSpamSubset.columns), enronSpamSubset.columns)
#print(len(completeSpamAssassin.columns), completeSpamAssassin.columns)

frames = [completeSpamAssassin, enronSpamSubset]

concat_frames = pd.concat(frames)

#print(len(concat_frames.columns), concat_frames.columns)


##Pre-procesamiento

# Eliminando NaN
concat_frames['Body'] = concat_frames['Body'].fillna('')

#print(concat_frames)

# Todas minusculas
concat_frames['Body'] = concat_frames['Body'].str.lower()

# Removiendo acentos
def removerAcentos(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ascii','ignore').decode('utf-8','ignore')
    return texto

# Removiendo caracteres especiales
def removerCaracteresEspecialesNumerosSimbolos(texto, removerDigitos =False):
    patron = r'[^a-zA-Z0-9\s]' if not removerDigitos else r'[^a-zA-Z\s]'
    texto = re.sub(patron,'', texto)
    return texto

body_prep = []
empty_rows = []
for i in range(len(concat_frames['Body'])):
    if len(concat_frames.iloc[i, 1]) == 5 and concat_frames.iloc[i, 1] == 'empty':
        concat_frames.iloc[i, 1] = ''
        empty_rows.append(i)

    i_sin_acentos = removerAcentos(concat_frames.iloc[i, 1].replace('\n', ' '))
    i_sin_acentos = i_sin_acentos.strip()
    i_sin_contracciones = contractions.fix(i_sin_acentos)
    i_sin_stopwords = remove_stopwords(i_sin_contracciones)
    body_prep.append(removerCaracteresEspecialesNumerosSimbolos(i_sin_stopwords))

concat_frames['Body'] = body_prep

#print(concat_frames)

# Metodo TF-IDF
tv = TfidfVectorizer(min_df=0.05, max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(concat_frames['Body'])
tv_matrix = tv_matrix.toarray()

vocabulario = tv.get_feature_names()
tf_idf = pd.DataFrame(np.round(tv_matrix, 2), columns=vocabulario)

final_df_tf_idf = concat_frames.join(tf_idf)
print(final_df_tf_idf)
final_df_tf_idf.to_csv('tf_idf_final_data.csv')

'''
# Metodo Bag of n-Grams

bv = CountVectorizer(ngram_range=(3,3))
bv_matrix = bv.fit_transform(concat_frames['Body'])

bv_matrix = bv_matrix.toarray()
vocabulario = bv.get_feature_names()
print(pd.DataFrame(bv_matrix, columns=vocabulario))
'''