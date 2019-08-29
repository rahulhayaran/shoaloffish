import gensim
from gensim.models import FastText
import pandas as pd
import numpy as np
import pickle as achar

def dump_achar(obj, filename):
    fileobj = open(filename, 'wb')
    achar.dump(obj, fileobj)
    fileobj.close()

tdf = pd.read_excel('timespent.xlsx')
adf = pd.read_excel('assessments.xlsx')
cdf = pd.read_excel('certification.xlsx')

rtdf = tdf['resource_name'].unique()
ctdf = tdf['course_name'].unique()
cadf = adf['course_name'].unique()
ccdf = cdf['certification_name'].unique()

total = list(rtdf) + list(ctdf) + list(cadf) + list(ccdf)

corpus = []

for t in total:
    corpus.append(t.split(' '))

model = FastText(size=300, window=3, min_count=1)
model.build_vocab(sentences=corpus)
model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

dump_achar(model.wv, 'fasttext_wv')