from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import string
import unicodedata
import re

start_number = re.compile(r"^ *\d+ *", re.IGNORECASE)

def normalizer(y):
    table = str.maketrans('','',string.punctuation)
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', y.lower().translate(table)) if not unicodedata.combining(c)))

def number_removal(y):
    return start_number.sub('', y)


def read(file):
    df = pd.read_csv(file)
    df['id'] = df.index
    df['proname'] = df['proname'].apply(normalizer).apply(number_removal)
    df['groname'] = df['groname'].apply(normalizer).apply(number_removal)
    df['feature_vector'] = df['groname'] + ' ' + df['proname']
    return df

documents = read('../bases/lilprobase.csv')

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents['feature_vector'].tolist())

af = AffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

print ('Clusters %d' % len(cluster_centers_indices))
print("\n")
print("Prediction")

Y = vectorizer.transform(["fanta maracujá"])
prediction = af.predict(Y)
print(documents.iloc[cluster_centers_indices[prediction[0]]]['groname'])

Y = vectorizer.transform(["pizza de beringela"])
prediction = af.predict(Y)
print(documents.iloc[cluster_centers_indices[prediction[0]]]['groname'])

