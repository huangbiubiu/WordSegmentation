import sklearn.feature_extraction
import numpy as np


class Ngram:
    def __init__(self, ngram_size):
        self.ngram_size: int = ngram_size
        self.vectorizer: sklearn.feature_extraction.text.CountVectorizer = sklearn.feature_extraction.text.CountVectorizer(
            ngram_range=(self.ngram_size, self.ngram_size))
        self.matrix = None

    def train(self, corpus: list) -> None:
        # corpus = corpus.replace("\n", " E ")
        self.matrix = self.vectorizer.fit_transform(corpus)
        np.sum(self.matrix, axis=0)

    def prob(self, sentence: str):
        return self.vectorizer.fit([sentence])

        pass

    pass


def main():
    path = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
    with open(path, encoding='utf8') as file:
        lines = file.readlines()

    model = Ngram(3)
    model.train(lines)

    prob = model.prob("从 根本上 解决 问题")
    pass


if __name__ == '__main__':
    main()
