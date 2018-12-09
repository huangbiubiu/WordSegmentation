import operator
from collections import deque

import util
from graph.GraphNode import GraphNode
from probability.Ngram import Ngram
from util import Constant


class DAG:
    ngram_size: int

    def __init__(self, ngram_size: int, sentence_len: int):
        self.start: GraphNode = GraphNode(Constant.START_SYMBOL, start=0, end=0)
        self.end: GraphNode = GraphNode(Constant.END_SYMBOL, start=sentence_len, end=sentence_len)

        self.start.accumulative_prob = 1

        self.ngram_size = ngram_size

        self.index_node = {0: {self.start.value: self.start},
                           sentence_len: {self.end.value: self.end}}

        pass

    def __add_nodes(self, node, sentence: str, start: int, end: int, word_dict: set):
        sub_sentence = sentence[start: end]
        if sub_sentence == Constant.END_SYMBOL:
            node.next.append(self.end)
            self.end.previous.append(node)
            return

        if start in self.index_node:
            next_dict: dict = self.index_node[start]
        else:
            next_dict: dict = {}
            self.index_node[start] = next_dict

        def add_next(index: int, word_: str):
            new_node = GraphNode(word_, start=start, end=start + index)
            next_dict[word_] = new_node
            node.add_next(new_node)
            self.__add_nodes(new_node, sentence, start=start + index, end=end, word_dict=word_dict)

        for i in range(1, len(sub_sentence)):
            word: str = sub_sentence[:i]
            if word in next_dict:
                node.add_next(next_dict[word])
            elif word in word_dict:
                add_next(i, word)

        # force to add next char as next word
        if node != self.end and len(node.next) == 0:
            add_next(1, sub_sentence[:1])
        pass

    def forward(self, probs: dict):
        self.__update_prob_recursive(self.start, probs, deque([self.start.value]))

    def backward(self) -> (list, float):
        """
        do backward on the DAG
        :return: maximum probability segmentation and its probability
        """
        result = deque()
        node: GraphNode = self.end
        accumulative_prob = 1
        while node != self.start:
            previous_node = node.best_previous
            accumulative_prob *= node.accumulative_prob

            node = previous_node
            result.appendleft(previous_node.value)

        return list(result), accumulative_prob

    def __update_prob_recursive(self, start: GraphNode, probs: dict, previous_words: deque):
        fixed_previous_words = previous_words
        for next_node in start.next:
            previous_words = fixed_previous_words.copy()
            pre_len = len(previous_words)

            next_node: GraphNode = next_node  # just for type declaration

            # prior probability
            prior_prob = probs[pre_len].probability(" ".join(previous_words))

            # union probability
            words = list(previous_words.copy())
            words.append(next_node.value)
            union_prob = probs[pre_len + 1].probability(" ".join(words))

            local_prob = union_prob / prior_prob
            accumulative_prob = local_prob * start.accumulative_prob

            if next_node.best_previous is None or accumulative_prob > next_node.accumulative_prob:
                next_node.accumulative_prob = accumulative_prob
                next_node.best_previous = start

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
        sentence = f"{Constant.START_SYMBOL}{sentence}{Constant.END_SYMBOL}"

        g = DAG(ngram_size, len(sentence))
        g.__add_nodes(g.start, sentence, start=1, end=len(sentence), word_dict=word_dict)

        return g


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    word_dict = util.construct_dict(dict_path)
    g = DAG.build_graph("去北京大学玩", word_dict)
    pass


if __name__ == '__main__':
    main()
