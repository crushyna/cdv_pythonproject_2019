import re

def isdate(input):
    return(bool(re.match(r"(\d{2}\S\d{2}\S\d{4})|(\d{2}\S\d{2}\S\d{2})", input)))

class DateRemover:
    def process(self, data):
        return " ".join(x for x in data.split() if not isdate(x))


        