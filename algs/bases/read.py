import pandas as pd
import numpy as np
import string
import unicodedata
import re
import json

start_number = re.compile(r"^ *\d+ *", re.IGNORECASE)

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
    return df

df = read('lilprobase.csv')

l = list(set(df['groname']))
l.sort()

f = open('classes.json', 'w')
f.write(json.dumps(l))
f.close()
