import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from util import Constant


class Ngram:
    def __init__(self, ngram_size, train_path: str):
        self.ngram_size: int = ngram_size
        self.vectorizer: CountVectorizer = CountVectorizer(
            ngram_range=(1, self.ngram_size + 1),
            token_pattern='\\b\\w+\\b')

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
        line = line.replace('\n', '')  # remove new line symbol
        line = f"{Constant.START_SYMBOL} {line} {Constant.END_SYMBOL}"
        return line

    def __train(self, corpus: list) -> csr_matrix:
        matrix: csr_matrix = self.vectorizer.fit_transform(corpus)
        matrix = matrix.sum(axis=0).astype(np.float32)

        matrix = matrix / matrix.sum()  # normalize the probability
        # TODO smoothing the prob

        return matrix

    def probability(self, word: str):
        if word in self.vectorizer.vocabulary_:
            index = self.vectorizer.vocabulary_[word]
            return self.matrix.A1[index]
        else:
            return 0


def main():
    path = "/home/hyh/projects/CLProject/WordSegmentation/data/train"

    model = Ngram(3, path)

    print(list(filter(lambda s: 'S' in s, model.vectorizer.vocabulary_.keys())))
    print(list(model.vectorizer.vocabulary_.keys())[:100])  # todo for debug

    pass


if __name__ == '__main__':
    main()
