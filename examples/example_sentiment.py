from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

# Vader is pretty easy to use. Let's use that!

test_sentence = "that was something they seemed really concerned about and they asked in both my interviews: \"Do you see yourself getting a graduate degree? What kind of degree would interest you most?\""

sentences = tokenize.sent_tokenize(test_sentence)

sid = SentimentIntensityAnalyzer()

for s in sentences:
    print(s)
    ss = sid.polarity_scores(s)

    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()
