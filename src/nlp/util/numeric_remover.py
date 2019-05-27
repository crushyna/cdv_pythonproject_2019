class NumericRemover:
    def process(self, data):                   #If not #isnumeric = True
        return " ".join(x for x in data.split() if not x.isnumeric())