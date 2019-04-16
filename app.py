from flask import Flask
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('punkt')



APP = Flask(__name__)
df2 = pd.read_csv('/content/drive/My Drive/simpsons_script_lines.csv',error_bad_lines=False)
#token_list = list()
#for i in df2['normalized_text']:
#  token_list.append(word_tokenize(str(i)))
#df2['token'] = token_list

@APP.route('/')
def hello_world():
    return df2['spoken_words'][0]