import regex as re

class PunctuationRemover:
    def process(self, data):
        return re.sub(r'[^\w\s]', ' ', data)