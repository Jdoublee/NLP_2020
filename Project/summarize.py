from lexrankr import LexRank

class Summarize:
    def __init__(self, paragraph, num):
        try:
            if isinstance(paragraph, str) is True:
                self.paragraph = paragraph
                self.probe_num = num
        except AttributeError as e:
            raise TypeError("You can't use it if it is not string.")

    def summarize(self):
        lex = LexRank()
        lex.summarize(self.paragraph)
        summaries = lex.probe(self.probe_num)
        return summaries