from operator import concat
from tkinter import EXCEPTION
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import unicodedata
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, model_selection


###Cargando datasets
df = pd.read_csv('./tf_idf_final_data.csv')
df = df.drop(['Body'], axis=1)

target = df['Label']
feature_matrix = df.drop(['Label'], axis=1)

print('Final features:', feature_matrix.columns)
feature_matrix.head()

#División de datos de entrenamientos y prueba
feature_matrix_train, feature_matrix_test, target_train, target_test = model_selection.train_test_split(feature_matrix, target, test_size=0.30, random_state=31)

#Aplicación de modelo
clf = MultinomialNB()
clf = clf.fit(feature_matrix_train, target_train)

#Guardando modelo
clf_pkl_model = open('model_tf_idf.pkl', 'wb')
pickle.dump(clf, clf_pkl_model)
clf_pkl_model.close()

print(feature_matrix_train.count())

print(feature_matrix_test.count())

target_pred = clf.predict(feature_matrix_test)

#Métricas
print(metrics.accuracy_score(target_test, target_pred))
print('Matriz de confusion /n',metrics.confusion_matrix(target_test, target_pred))
print(metrics.classification_report(target_test, target_pred, target_names=['spam', 'legitimate']))