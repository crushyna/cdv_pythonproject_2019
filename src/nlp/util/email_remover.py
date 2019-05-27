import re

def isemail(input):
    return(bool(re.match(r"\w+@\w+\W\w{2,3}", input)))

class EmailRemover:
    def process(self, data):
        return " ".join(x for x in data.split() if not isemail(x))