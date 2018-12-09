import os.path

import util
from dataset.Preprocess import process_data
from graph.DAG import DAG
from probability.Ngram import Ngram
import pickle


class ProbCalculator:
    __NGRAM_PATH: str = 'cache/ngram.pkl'
    __DICT_PATH: str = 'cache/dict.pkl'

    def __init__(self, dict_path: str, corpus_path: str, ngram_size: int):
        self.ngram_size = ngram_size

        if os.path.exists(self.__NGRAM_PATH):
            print(f'load saved ngram')
            with open(self.__NGRAM_PATH, 'rb') as file:
                self.ngram = pickle.load(file)
        else:
            print(f"building {ngram_size}-gram language model")
            self.ngram = Ngram(ngram_size, corpus_path)

            if not os.path.exists('cache/'):
                os.mkdir('cache/')
            print('saving ngram to disk')
            with open(self.__NGRAM_PATH, 'wb') as file:
                pickle.dump(self.ngram, file)

        if os.path.exists(self.__DICT_PATH):
            print(f'load saved word dict')
            with open(self.__DICT_PATH, 'rb') as file:
                self.word_dict = pickle.load(file)
        else:
            print("building word dict")
            self.word_dict = util.construct_dict(dict_path)

            print('saving dict to disk')
            with open(self.__DICT_PATH, 'wb') as file:
                pickle.dump(self.word_dict, file)

        print("initialization completed.")

    def calc(self, sentence):
        # TODO error when sentence contains ENGLISH WORDS and DIGITS
        if isinstance(sentence, str):
            original_sentence = sentence
            # sentence = process_data(sentence, self.word_dict)

            g = DAG.build_graph(sentence, self.word_dict, ngram_size=self.ngram_size)

            g.forward(self.ngram)
            segment, _ = g.backward()

            return " ".join(segment)[2:]  # remove start symbol
        elif isinstance(sentence, list):
            return list(map(self.calc, sentence))
        else:
            raise TypeError(f"Type {type(sentence)} is not supported.")


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    corpus_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
    graph = ProbCalculator(dict_path, corpus_path, ngram_size=2)
    g = graph.calc("去北京大学玩")

    print(g)

    pass


if __name__ == '__main__':
    main()
