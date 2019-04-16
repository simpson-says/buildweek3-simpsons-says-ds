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




@APP.route('/')
def hello_world():

    user_input = "it was a little of both. Sometimes when a disease is in all the magazines"
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


@APP.route('/quoteretieve')
def root():
    result = """
    [{'quote_id': 9549,
  'raw_character_text': 'Miss Hoover',
  'spoken_words': "No, actually, it was a little of both. Sometimes when a disease is in all the magazines and all the news shows, it's only natural that you think you have it.",
  'episode_title': "Lisa's Substitute",
  'season': 2,
  'number_in_season': 19},
 {'quote_id': 9550,
  'raw_character_text': 'Lisa Simpson',
  'spoken_words': "Where's Mr. Bergstrom?",
  'episode_title': "Lisa's Substitute",
  'season': 2,
  'number_in_season': 19},
 {'quote_id': 9551,
  'raw_character_text': 'Miss Hoover',
  'spoken_words': "I don't know. Although I'd sure like to talk to him. He didn't touch my lesson plan. What did he teach you?",
  'episode_title': "Lisa's Substitute",
  'season': 2,
  'number_in_season': 19},
 {'quote_id': 9552,
  'raw_character_text': 'Lisa Simpson',
  'spoken_words': 'That life is worth living.',
  'episode_title': "Lisa's Substitute",
  'season': 2,
  'number_in_season': 19}]
    """

    return result


@APP.route('/search')
def search():
    result = """
    {'quote_id': 9552,
  'raw_character_text': 'Lisa Simpson',
  'spoken_words': 'That life is worth living.',
  'episode_title': "Lisa's Substitute",
  'season': 2,
  'number_in_season': 19}
    """

    return result