import os.path
import pickle

import util
from dataset.Preprocess import process_data
from graph.DAG import DAG
from probability.Ngram import Ngram


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
            self.ngram = {}
            for n in range(1, ngram_size + 1):
                print(f"building {n}-gram language model")
                self.ngram[n] = Ngram(n, corpus_path)

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
        if isinstance(sentence, str):
            original_sentence = sentence
            sentence: str = process_data(sentence, self.word_dict)[0]

            segments = []
            for s in sentence.split():
                g = DAG.build_graph(s, self.word_dict, ngram_size=self.ngram_size)

                g.forward(self.ngram)
                segment, _ = g.backward()

                split_result = " ".join(segment)[2:]

                segments.append(split_result)

            return " ".join(segments)  # remove start symbol
        elif isinstance(sentence, list):
            return list(map(self.calc, sentence))
        else:
            raise TypeError(f"Type {type(sentence)} is not supported.")


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    corpus_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
    graph = ProbCalculator(dict_path, corpus_path, ngram_size=2)
    g = graph.calc("中华人民共和国今天成立啦")
    # g = graph.calc("去北京大学玩")

    print(g)

    pass


if __name__ == '__main__':
    main()
