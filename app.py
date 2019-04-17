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
df = df.rename(columns={'spoken_words_x':'spoken_words','raw_character_text_x':'raw_character_text'})
#df3 = pickle.load( open( "scripts.pkl", "rb" ) )
corpus = pickle.load( open( "c.p", "rb" ) )
#s = gensim.similarities.Similarity('/app/',t[corpus],num_features=len(d))
#pickle.dump(s,open('sims2.p','wb'))

print(df.head())


@APP.route('/')
@APP.route('/api',methods=['POST'])
def hello_world():

    user_input = "the goggles do nothing"
    if request.method == 'POST':
        user_input = request.values['quote']
    print(user_input)
    query_doc = [w.lower() for w in word_tokenize(user_input)]
    print(query_doc)
    query_doc_bow = d.doc2bow(query_doc)
    query_doc_tf_idf = t[query_doc_bow]
    v = s[query_doc_tf_idf]
    result = v.argsort()[-10:][::-1]
    condition = (df.index.isin(result))
    response = df[condition]
    column = ['quote_id', 'raw_character_text', 'spoken_words','episode_title','season','number_in_season']
    response = response[column]
    response.to_json(orient='records')
    print(response)


    return response.to_json(orient='records')

@APP.route('/getquote',methods=['POST'])
@APP.route('/getquote')
def getquote():
    inputs = '[1,2,3]'
    if request.method=='POST':
        inputs = request.values['input']
    inputs2 = [int(x) for x in inputs.strip('[]').split(',')]

    #l =[9560, 41110, 76160, 76216, 105073]
    condition = (df.quote_id.isin(inputs2))
    response = df[condition]
    column = ['quote_id', 'raw_character_text', 'spoken_words','episode_title','season','number_in_season']
    response = response[column]
    response.to_json(orient='records')
    print(response)


    return response.to_json(orient='records')




