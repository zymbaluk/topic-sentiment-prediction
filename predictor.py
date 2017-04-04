from gensim import models, corpora
import pickle
from gensim_helpers import *
from pdb import set_trace
from collections import defaultdict





lda_location = "trained_models/lda"
dictionary_location = "trained_models/dicc"
sentiment_tracker_location = "trained_models/st.pickle"

lda = models.LdaModel.load(lda_location)
dictionary = corpora.Dictionary.load(dictionary_location)
sentiment_tracker = None

with open('trained_models/st.pickle', 'rb') as inp:
    sentiment_tracker = pickle.load(inp)

topic_sentiment = dict()

#here we find the sentiment as used in the corpus, and average it out for the topic
#we trained our model with 100 topics, go thru all of them
for topic_num in range(100):
    tmp_sent = defaultdict(float)
    #cycle through the top 20 most significant words in each topic
    for word, weight in lda.show_topic(topic_num, topn=20):
        
        try:
            old_sentiment = sentiment_tracker.avg_sent[word]

            for key in old_sentiment.keys():
                tmp_sent[key] += (old_sentiment[key]*weight)

        except KeyError:
            #get the sentiment of the word as it appeared in the corpus
            tmp_sent = sentiment_tracker.avg_sent[word]

            for key in sent.keys():
                #multiply each sentiment value by the weight of the word in the topic
                tmp_sent[key] *= weight

    topic_sentiment[topic_num] = tmp_sent

with open('trained_models/topic_sentiment.dict', 'wb') as out:
    pickle.dump(topic_sentiment, out, pickle.HIGHEST_PROTOCOL)