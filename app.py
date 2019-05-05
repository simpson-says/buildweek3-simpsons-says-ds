from flask import Flask , request, make_response
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import gensim
import random
import numpy as np
import json

APP = Flask(__name__)

d = pickle.load( open( "dictionary.p", "rb" ) )
t = pickle.load( open( "tf_idf.p", "rb" ) )
s = pickle.load( open( "sims2.p", "rb" ) )
df = pickle.load( open( "df.p", "rb" ) )
df = df.rename(columns={'spoken_words_x':'spoken_words','raw_character_text_x':'raw_character_text'})
corpus = pickle.load( open( "c.p", "rb" ) )
quote_dump = pickle.load(open("quote_dump.pkl", "rb" ))
## Uncomment and run once for local operation
# s = gensim.similarities.Similarity('/app/',t[corpus],num_features=len(d))
# pickle.dump(s,open('sims2.p','wb'))


@APP.route('/')
@APP.route('/api', methods=['POST'])
def hello_world():

    user_input = "the goggles do nothing"
    if request.method == 'POST':
        user_input = request.values['quote']
    query_doc = [w.lower() for w in word_tokenize(user_input)]
    query_doc_bow = d.doc2bow(query_doc)
    query_doc_tf_idf = t[query_doc_bow]
    v = s[query_doc_tf_idf]
    result = v.argsort()[-10:][::-1]
    condition = (df.index.isin(result))
    response = df[condition]
    column = ['quote_id', 'raw_character_text', 'spoken_words','episode_title','season','number_in_season']
    response = response[column]
    response.to_json(orient='records')

    return response.to_json(orient='records')


@APP.route('/getquote',methods=['POST'])
@APP.route('/getquote')
def getquote():
    inputs = '[1,2,3]'
    if request.method=='POST':
        inputs = request.get_json(force=True)['input']
    
    condition = (df.quote_id.isin(inputs))
    response = df[condition]
    print(response)
    column = ['quote_id', 'raw_character_text', 'spoken_words','episode_title','season','number_in_season']
    response = response[column]
    response.to_json(orient='records')

    return response.to_json(orient='records')


@APP.route('/gen', methods=['POST'])
@APP.route('/gen')
def generator():
    # Acceptable inputs = ['homer', 'marge', 'bart', 'lisa', 'moe', 'grampa', 'skinner']
    name = 'homer'
    if request.method=='POST':
        name = request.values['input']
    
    rand_quotes = random.choices(quote_dump[name], k=10)
    quotes2 = [{'charname':name, 'quote':x} for x in rand_quotes]
    return_list = json.dumps(quotes2)
    return return_list


