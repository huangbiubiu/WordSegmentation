import sklearn.feature_extraction
from scipy.sparse import csr_matrix
import numpy as np


class Ngram:
    def __init__(self, ngram_size, train_path: str):
        self.ngram_size: int = ngram_size
        self.vectorizer: sklearn.feature_extraction.text.CountVectorizer = sklearn.feature_extraction.text.CountVectorizer(
            ngram_range=(self.ngram_size, self.ngram_size))

        corpus: list = self.__load_corpus(train_path)
        self.matrix: csr_matrix = self.__train(corpus)

    @staticmethod
    def __load_corpus(path: str) -> list:
        with open(path, encoding='utf8') as file:
            lines = file.readlines()

        lines = list(map(Ngram.process_line, lines))  # remove whitespaces in corpus

        return lines

    @staticmethod
    def process_line(line: str) -> str:
        line = f"S {line} E"
        return line

    def __train(self, corpus: list) -> csr_matrix:
        matrix: csr_matrix = self.vectorizer.fit_transform(corpus)
        matrix = matrix.sum(axis=0).astype(np.float32)

        matrix = matrix / matrix.sum()  # normalize the probability
        # TODO smoothing the prob

        return matrix

    def probability(self, word: str):
        # if len(word.split()) != self.ngram_size:
        #     raise ValueError(f"Sentence length should be {self.ngram_size}")
        if word in self.vectorizer.vocabulary_:
            index = self.vectorizer.vocabulary_[word]
            return self.matrix[index]
        else:
            return 0


def main():
    path = "/home/hyh/projects/CLProject/WordSegmentation/data/train"

    model = Ngram(3, path)

    print(list(model.vectorizer.vocabulary_.keys())[:100])  # todo for debug

    pass


if __name__ == '__main__':
    main()
