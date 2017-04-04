import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim_helpers import *
import pdb
import pickle
import string

u = utility_helper()
dicc = corpora.Dictionary()
m = mem_lite_iter("../RC_2007-03.bz2")
sid = SentimentIntensityAnalyzer()
st = sentiment_tracker()

for s in m:
    sentence_sentiment = sid.polarity_scores(s)
    treated_text = treat_text(s, u)
    dicc.add_documents(documents=[treated_text])

    for word in treated_text:
        st.update(word, sentence_sentiment)

dicc.save('trained_models/dicc')

with open('trained_models/st.pickle', 'wb') as output:
    pickle.dump(st, output, pickle.HIGHEST_PROTOCOL)

# pdb.set_trace()

corpus = corpus_mem_lite(m, dicc)

ldamodel = models.LdaModel(corpus, num_topics=100, id2word=dicc, passes=20)

ldamodel.save('trained_models/lda')

pdb.set_trace()