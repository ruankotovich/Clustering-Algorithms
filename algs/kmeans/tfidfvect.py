from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string
import unicodedata
import re
import sys

start_number = re.compile(r"\d+", re.IGNORECASE)


def normalizer(y):
    table = str.maketrans('','',string.punctuation)
    return re.sub('(  +|^ +| +$)','',unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', y.lower().translate(table)) if not unicodedata.combining(c))).lower())

def number_removal(y):
    return start_number.sub('', y)


def read(file):
    df = pd.read_csv(file)
    df['id'] = df.index
    df['proname'] = df['proname'].apply(normalizer).apply(number_removal)
    df['groname'] = df['groname'].apply(normalizer).apply(number_removal)
    df['feature_vector'] = df['groname'].apply(normalizer).apply(number_removal) + ' ' + df['proname'].apply(normalizer).apply(number_removal)
    return df


documents = read('../bases/lilprobase.csv')

vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
X = vectorizer.fit_transform(documents['feature_vector'].tolist())

true_k = int(sys.argv[1])
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

# Nice Pythonic way to get the indices of the points for each corresponding cluster
mydict = {i: np.where(model.labels_ == i)[0] for i in range(model.n_clusters)}

# Transform this dictionary into list (if you need a list as result)
# dictlist = []
# for key, value in mydict.items():
# temp = [key,value]
# dictlist.append(temp)

# print("Dictlist:")
# print(dictlist)

# print("Top terms per cluster:")
documents['group'] = model.labels_
print("Groups:")
print(documents.sort_values('group'))

file = open('knowledgebase.cpp_part','w')

cur = {'group': -1, 'groupPieceName': [], 'clusterName': '',
       'tagName': '', 'productPieceName': []}
for i, doc in documents.sort_values('group').T.iteritems():
    if doc['group'] != cur['group']:
        file.write(
            '''
            absClusters.emplace_back("{groupNames}",
                                     "{groupName}",
                                     "{tagName}");
            absClusters.back().bulkFeed({{
                {productNames}
            }});
            '''.format(
                    groupNames=' '.join(set(cur['groupPieceName'])), 
                    groupName=cur['clusterName'],
                    tagName=cur['tagName'],
                    productNames=',\n\t\t'.join( set(list(map(lambda x: '"{x}"'.format(x=x), cur['productPieceName']))) )
                )
        )

        cur['group'] = doc['group']
        cur['groupPieceName'] = [doc['groname']]
        cur['productPieceName'] = [doc['proname']]
        cur['clusterName'] = doc['groname']
        cur['tagName'] = doc['groname']
    else:
        cur['groupPieceName'] += [doc['groname']]
        cur['productPieceName'] += [doc['proname']]

file.close()
