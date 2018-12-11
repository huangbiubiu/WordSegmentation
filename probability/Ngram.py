import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from util import Constant


class Ngram:
    matrix: csr_matrix
    total: int

    def __init__(self, ngram_size, train_path: str):
        self.ngram_size: int = ngram_size

        self.vectorizer: CountVectorizer = CountVectorizer(
            ngram_range=(self.ngram_size, self.ngram_size),
            token_pattern='\\b\\w+\\b')

        corpus: list = self.__load_corpus(train_path)
        self.matrix, self.total = self.__train(corpus)

    @staticmethod
    def __load_corpus(path: str) -> list:
        with open(path, encoding='utf8') as file:
            lines = file.readlines()

        lines = list(map(Ngram.process_line, lines))  # remove whitespaces in corpus

        return lines

    @staticmethod
    def __read_vocabulary(path="../data/dict.txt.small"):
        with open(path, encoding='utf8') as file:
            vocab = file.readlines()
        vocab = list(map(lambda s: s.split()[0], vocab))

        return vocab

    @staticmethod
    def process_line(line: str) -> str:
        line = line.replace('\n', '')  # remove new line symbol
        line = f"{Constant.START_SYMBOL} {line} {Constant.END_SYMBOL}"
        return line

    def __train(self, corpus: list) -> (csr_matrix, int):
        matrix: csr_matrix = self.vectorizer.fit_transform(corpus)
        matrix = matrix.sum(axis=0).astype(np.float32)

        return matrix, matrix.sum()

    def probability(self, word: str):
        if word in self.vectorizer.vocabulary_:
            index = self.vectorizer.vocabulary_[word]
            return self.matrix.A1[index] / self.total
        else:
            return 1 / self.total  # laplace smoothing, not good enough


def main():
    path = "/home/hyh/projects/CLProject/WordSegmentation/data/train"

    model = Ngram(3, path)

    print(list(filter(lambda s: 'S' in s, model.vectorizer.vocabulary_.keys())))
    print(list(model.vectorizer.vocabulary_.keys())[:100])  # todo for debug

    pass


if __name__ == '__main__':
    main()
