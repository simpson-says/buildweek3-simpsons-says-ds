from flask import Flask , request
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('punkt')
import pickle
import gensim

import os

dirpath = os.getcwd()
dirpath = dirpath +'/'


APP = Flask(__name__)

d = pickle.load( open( "dictionary.p", "rb" ) )
t = pickle.load( open( "tf_idf.p", "rb" ) )
s = pickle.load( open( "sims2.p", "rb" ) )
df = pickle.load( open( "df.p", "rb" ) )
corpus = pickle.load( open( "c.p", "rb" ) )
#sims = gensim.similarities.Similarity("/app/",t[corpus],num_features=len(d))
#pickle.dump(sims,open('sims2.p','wb'))




@APP.route('/',methods=['POST'])
def hello_world():

    user_input = "testing"
    query_doc = [w.lower() for w in word_tokenize(user_input)]
    print(query_doc)
    query_doc_bow = d.doc2bow(query_doc)
    query_doc_tf_idf = t[query_doc_bow]
    v = s[query_doc_tf_idf]
    result = v.argsort()[-10:][::-1]
    condition = (df.index.isin(result))
    reponse = df[condition]
    # reponse.to_json()
    print(reponse)


    return reponse.to_json()