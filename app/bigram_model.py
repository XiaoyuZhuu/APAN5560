import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        self.model = defaultdict(list)
        self.build_model(corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words)-1):
                self.model[words[i]].append(words[i+1])

    def generate_text(self, start_word, length=10):
        word = start_word.lower()
        result = [word]
        for _ in range(length-1):
            if word in self.model:
                word = random.choice(self.model[word])
            else:
                break
            result.append(word)
        return " ".join(result)
