from collections import deque

import util
from graph.GraphNode import GraphNode
from util import Constant


class DAG:
    ngram_size: int

    def __init__(self, ngram_size: int):
        self.start: GraphNode = GraphNode(Constant.START_SYMBOL)
        self.end: GraphNode = GraphNode(Constant.END_SYMBOL)

        self.ngram_size = ngram_size

    def __add_nodes(self, node, sub_sentence: str, word_dict: set):
        if sub_sentence == Constant.END_SYMBOL:
            node.next.append(self.end)
            self.end.previous.append(node)
            return

        for i in range(1, len(sub_sentence)):
            word: str = sub_sentence[:i]
            if word in word_dict:
                new_node = GraphNode(word)
                node.add_next(new_node)
                self.__add_nodes(new_node, sub_sentence[i:], word_dict)
        pass

    def forward(self, probs: dict):
        self.__update_prob_recursive(self.start, probs, deque([self.start.value]))

    def __update_prob_recursive(self, start: GraphNode, probs: dict, previous_words: deque):


        for next_node in start.next:
            next_node: GraphNode = next_node  # just for type declaration
            pre_len = len(previous_words)

            # prior probability
            prior_prob = probs[pre_len].probability(" ".join(previous_words))

            # union probability
            words = list(previous_words.copy())
            words.append(next_node.value)
            union_prob = probs[pre_len + 1].probability(" ".join(words))

            # update accumulative probability
            if prior_prob == 0:
                accumulative_prob = 0
            else:
                accumulative_prob = union_prob / prior_prob
            next_node.accumulative_prob[start] = accumulative_prob

            if start == self.end:
                start.next.append(self.end)
                self.end.previous.append(start)
                return

            # update previous words and continue recursion
            previous_words.append(next_node.value)
            if len(previous_words) >= self.ngram_size:
                previous_words.popleft()
            self.__update_prob_recursive(next_node, probs, previous_words)

    @staticmethod
    def build_graph(sentence: str, word_dict: set, ngram_size: int):
        sentence = f"{sentence}{Constant.END_SYMBOL}"

        g = DAG(ngram_size)
        g.__add_nodes(g.start, sentence, word_dict)

        return g


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    word_dict = util.construct_dict(dict_path)
    g = DAG.build_graph("去北京大学玩", word_dict)
    pass


if __name__ == '__main__':
    main()
