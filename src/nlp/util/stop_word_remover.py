import nltk

class StopWordRemover:
    def process(self, data):
        stop = nltk.corpus.stopwords.words('english')
        return ' '.join(x for x in data.split() if x not in stop)