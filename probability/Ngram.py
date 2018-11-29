import sklearn.feature_extraction
from scipy.sparse import csr_matrix
import numpy as np


class Ngram:
    def __init__(self, ngram_size):
        self.ngram_size: int = ngram_size
        self.vectorizer: sklearn.feature_extraction.text.CountVectorizer = sklearn.feature_extraction.text.CountVectorizer(
            ngram_range=(self.ngram_size, self.ngram_size))
        self.matrix: csr_matrix = None

    def train(self, corpus: list) -> None:
        self.matrix: csr_matrix = self.vectorizer.fit_transform(corpus)
        self.matrix = self.matrix.sum(axis=0).astype(np.float32)

        self.matrix = self.matrix / self.matrix.sum()  # normalize the probability
        # TODO smoothing the prob

    def probability(self, word: str):
        if len(word.split()) != self.ngram_size:
            raise ValueError(f"Sentence length should be {self.ngram_size}")
        if word in self.vectorizer.vocabulary_:
            index = self.vectorizer.vocabulary_[word]
            return self.matrix[index]
        else:
            return 0


def main():
    path = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
    with open(path, encoding='utf8') as file:
        lines = file.readlines()

    model = Ngram(3)
    model.train(lines)

    print(list(model.vectorizer.vocabulary_.keys())[:100])  # todo for debug

    pass


if __name__ == '__main__':
    main()
