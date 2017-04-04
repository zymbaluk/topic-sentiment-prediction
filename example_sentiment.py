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

class corpus_mem_lite(object):
    def __init__(self, _m, _d):
        self.m = _m
        self.d = _d

    def __iter__(self):
        for text in self.m:
            yield self.d.doc2bow(text.lower().split())

def treat_text(sentence, utility):
    # tokenize into words by converting to lower-case and splitting by white space
    tokens = word_tokenize("".join(ch for ch in sentence.lower() if ch not in string.punctuation))
    
    # remove stop-words from tokens
    stopped_tokens = [i for i in tokens if i not in utility.stop_words and len(i)<=20]

    # stem tokens using Porter stemmer algorithm
    stemmed_tokens = [utility.stem(i) for i in stopped_tokens]

    return stemmed_tokens

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