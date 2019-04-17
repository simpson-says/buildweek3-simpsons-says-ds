from flask import Flask , request
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import gensim


APP = Flask(__name__)

d = pickle.load( open( "dictionary.p", "rb" ) )
t = pickle.load( open( "tf_idf.p", "rb" ) )
s = pickle.load( open( "sims2.p", "rb" ) )
df = pickle.load( open( "df.p", "rb" ) )
df = df.rename(columns={'spoken_words_x':'spoken_words','raw_character_text_x':'raw_character_text'})
corpus = pickle.load( open( "c.p", "rb" ) )

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


# @APP.route('/gen',methods=['POST'])
# @APP.route('/gen')
# def generator():
#     inputs = '[1,2,3]'
#     if request.method=='POST':
#         inputs = request.values['input']
    
#     placeholder = """
# (re: gown) hmmm... the sea won't stand for this.
# i know! i'll do a rap.(beat box noise)
# (re: beaver dam) see? animals can you?
# (reading) failure to wait by you can never
# (re: car
# i need some supplies: a keg of beer(he stands and says:) after sex
# (re: car
# angle on: greeting card
# (re: clocks) look at those celebrities: of all of the sea lion.(terrified)
# and here's the greatest heroes of all the...
# i know! i'll do a rap.(beat box noise)
# (re: beaver dam) see? animals can you?
# (reading) failure to wait by you can never
# (re: car
# i need some supplies: a keg of beer(he stands and says:) after sex
# (re: car
# angle on: greeting card
# (re: clocks) look at those celebrities: of all of the sea lion.(terrified)
# and here's the greatest heroes of all the...
# i know! i'll do a rap.(beat box noise)
# (re: beaver dam) see? animals can you?
# (reading) failure to wait by you can never
# (re: car
# """
    
#     return placeholder


