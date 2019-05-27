from src.nlp.util.lemmatizer import Lemmatizer
from src.nlp.util.punctuation_remover import PunctuationRemover
from src.nlp.util.numeric_remover import NumericRemover
from src.nlp.util.lowercase_applier import LowercaseApplier
from src.nlp.util.stop_word_remover import StopWordRemover
from src.nlp.util.hashtag_remover import HashtagRemover
from src.nlp.util.date_remover import DateRemover
from src.nlp.util.email_remover import EmailRemover

class TextProcessor:

    punctuationRemover = None
    numericRemover = None
    lowercaseApplier = None
    stopWordRemover = None
    lemmatizer = None
    hashtagRemover = None
    dateRemover = None
    emailRemover = None

    def __init__(self):
        self.punctuationRemover = PunctuationRemover()
        self.numericRemover = NumericRemover()
        self.lowercaseApplier = LowercaseApplier()
        self.stopWordRemover = StopWordRemover()
        self.lemmatizer = Lemmatizer()
        self.hashtagRemover = HashtagRemover()
        self.dateRemover = DateRemover()
        self.emailRemover = EmailRemover()

    def normalize(self, data):
        data = self.numericRemover.process(data)
        data = self.punctuationRemover.process(data)
        data = self.lowercaseApplier.process(data)
        data = self.stopWordRemover.process(data)
        data = self.lemmatizer.process(data)
        data = self.hashtagRemover.process(data)
        data = self.dateRemover.process(data)
        data = self.emailRemover.process(data)
        return data
