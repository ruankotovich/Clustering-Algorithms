from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
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


tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(documents['feature_vector'].tolist())]

max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")