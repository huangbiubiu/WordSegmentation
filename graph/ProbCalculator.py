import util
from graph.DAG import DAG
from probability.Ngram import Ngram


class ProbCalculator:
    def __init__(self, dict_path: str, corpus_path: str, ngram_size: int):
        self.ngram_size = ngram_size

        self.ngram = {}
        for n in range(1, ngram_size):
            print(f"building {n}-gram language model")
            self.ngram[n] = Ngram(n, corpus_path)

        print("building word dict")
        self.word_dict = util.construct_dict(dict_path)

        print("initialization completed.")

    def calc(self, sentence):
        g = DAG.get_graph(sentence, self.word_dict)

        route = dict()
        n = len(sentence)

        for n in range(1, n):

            pass

        return route


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    corpus_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
    graph = ProbCalculator(dict_path, corpus_path, 3)
    g = graph.calc("去北京大学玩")

    print(g)

    pass


if __name__ == '__main__':
    main()
