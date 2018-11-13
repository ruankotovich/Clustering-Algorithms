from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import string
import unicodedata
import re

start_number = re.compile(r"^ *\d+ *", re.IGNORECASE)
embedding = KeyedVectors.load_word2vec_format('~/Documents/Huge/skip_s50.txt')

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

def phrase2vec(ph):
        vec = None
        for word in ph:
                vec = embedding[word] if not vec else vec + model[word]
        return vec

documents = read('../bases/lilprobase.csv')

# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(documents['feature_vector'].tolist())
X = list(map(lambda x: phrase2vec(x), documents['feature_vector'].tolist()))
print(X)

true_k = 100
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

# Nice Pythonic way to get the indices of the points for each corresponding cluster
mydict = {i: np.where(model.labels_ == i)[0] for i in range(model.n_clusters)}

# Transform this dictionary into list (if you need a list as result)
dictlist = []
for key, value in mydict.items():
    temp = [key,value]
    dictlist.append(temp)

print("Dictlist:")
print(dictlist)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
featuredterm = {}
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :1]:
        print(' %s' % terms[ind]),
        featuredterm[i] = terms[ind]
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["fanta maracujá"])
prediction = model.predict(Y)
print(featuredterm[prediction[0]])

Y = vectorizer.transform(["pizza de beringela"])
prediction = model.predict(Y)
print(featuredterm[prediction[0]])


