from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import string
import unicodedata
import re

start_number = re.compile(r"^ *\d+ *", re.IGNORECASE)
embbed = Doc2Vec.load("d2v.model")

def normalizer(y):
    table = str.maketrans('','',string.punctuation)
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', y.lower().translate(table)) if not unicodedata.combining(c)))

def number_removal(y):
    return start_number.sub('', y)

def phrase2vec(p):
    return embbed.infer_vector(number_removal(normalizer(p)))

def read(file):
    df = pd.read_csv(file)
    df['id'] = df.index
    df['proname'] = df['proname'].apply(normalizer).apply(number_removal)
    df['groname'] = df['groname'].apply(normalizer).apply(number_removal)
    df['feature_vector'] = df['groname'] + ' ' + df['proname']
    return df

documents = read('../bases/lilprobase.csv')

X = list(map(lambda x: embbed.infer_vector(x), documents['feature_vector'].tolist()))
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(documents['feature_vector'].tolist())

true_k = 100
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

documents['cluster'] = model.labels_
print(documents);

print("Prediction")

Y = [phrase2vec("fanta maracujá")]
print(Y)
prediction = model.predict(Y)
print(documents[documents['cluster'] == prediction[0]])

print("\n")

Y = [phrase2vec("pizza de beringela")]
print(Y)
prediction = model.predict(Y)
print(documents[documents['cluster'] == prediction[0]])