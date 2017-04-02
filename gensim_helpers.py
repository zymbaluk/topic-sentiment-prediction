import bz2, json
from nltk import tokenize
from gensim import corpora
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

class mem_lite_iter(object):
    def __init__(self, _file_name):
        self.file_name = _file_name

    def __iter__(self):
        for line in bz2.open(self.file_name, mode='rt'):
            yield json.loads(line)['body']

#contains initialized objects we need. Pass this around instead of all of them
class utility_helper(object):
    
    def __init__(self):
        self.stop_words = get_stop_words('en')
        self.stemmer = PorterStemmer()

    def stem(self, tokens):
        return self.stemmer.stem(tokens)