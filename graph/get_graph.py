import numpy as np
from scipy.sparse import csc_matrix

from probability.Ngram import Ngram
import util


class Graph:
    def __init__(self, dict_path: str, corpus_path: str, ngram_size: int):
        self.ngram_size = ngram_size

        print(f"building {ngram_size}-gram language model")
        self.ngram = Ngram(ngram_size, corpus_path)

        print("building word dict")
        self.word_dict = util.construct_dict(dict_path)

        print("initialization completed.")

    def get_graph(self, sentence):
        sentence = f"S {sentence} E"

        graph = csc_matrix(np.eye(len(sentence)))
        for i in range(len(sentence)):
            for j in range(i + 1, len(sentence)):
                word = sentence[i:j + 1]
                if word in self.word_dict:
                    graph[i, j] = True
        return graph



    @staticmethod
    def in_nodes(index: int, graph) -> list:
        return graph[:, index]

    @staticmethod
    def out_node(index: int, graph) -> list:
        return graph[index, :]

    @staticmethod
    def get_previous_n(current_index: int, n: int, graph) -> list:
        previous = [[i] for i in list(Graph.in_nodes(current_index, graph))]
        for _ in range(n - 1):
            new_previous = []
            for path in previous:
                for p in Graph.in_nodes(path[-1], graph):
                    new_previous.append(path.append(p))

        return previous

    def calc(self, sentence):
        graph = self.get_graph(sentence)

        route = dict()
        n = len(sentence)

        for n in range(1, n):

            pass

        return route


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    corpus_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train"
    graph = Graph(dict_path, corpus_path, 3)
    g = graph.calc("去北京大学玩")

    print(g)

    pass


if __name__ == '__main__':
    main()
