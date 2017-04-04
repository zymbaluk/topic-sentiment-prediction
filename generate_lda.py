import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim_helpers import mem_lite_iter, utility_helper, sentiment_tracker
import pdb
import pickle
import string
from multiprocessing import freeze_support

class corpus_mem_lite(object):
    def __init__(self, _m, _d):
        self.m = _m
        self.d = _d

    def __iter__(self):
        for text in self.m:
            yield self.d.doc2bow(text.lower().split())

dicc = corpora.Dictionary.load('trained_models/dicc')

m = mem_lite_iter("../RC_2007-03.bz2")

corpus = corpus_mem_lite(m, dicc)

ldamodel = models.LdaModel(corpus, num_topics=100, id2word=dicc, passes=20)

ldamodel.save('trained_models/lda')