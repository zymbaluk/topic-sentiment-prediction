import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim_helpers import mem_lite_iter, utility_helper
import pdb


def treat_text(sentence, utility):
    # tokenize into words by converting to lower-case and splitting by white space
    tokens = s.lower().split()
    
    # remove stop-words from tokens
    stopped_tokens = [i for i in tokens if i not in utility.stop_words]

    # stem tokens using Porter stemmer algorithm
    stemmed_tokens = [utility.stem(i) for i in stopped_tokens]

    return stemmed_tokens


test_sent = """Uh this is actually pretty awesome, the driver was super cool about it as well. What a nice change of pace from the random spam texts and all that, super cool of both of them to be so chill.
Glad they were only texting when traffic was at a stand-still, I love all your beautiful faces and hope you all live long and happy lives which is why driving deserves your full attention while you're on the move."""

sentences = tokenize.sent_tokenize(test_sent)

u = utility_helper()

dicc = corpora.Dictionary()

m = mem_lite_iter("../RC_2005-12.bz2")

for s in m:
    treated_text = treat_text(s, u)
    dicc.add_documents(documents=[treated_text])

dicc.save("dicc.dic")