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

        for i in range(1, len(sub_sentence)):
            word: str = sub_sentence[:i]
            if word in next_dict:
                node.add_next(next_dict[word])
            elif word in word_dict:
                new_node = GraphNode(word, start=start, end=start + i)
                next_dict[word] = new_node
                node.add_next(new_node)
                self.__add_nodes(new_node, sentence, start=start + i, end=end, word_dict=word_dict)
        pass

    def forward(self, probs: Ngram):
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
            previous_node = max(node.accumulative_prob.items(), key=operator.itemgetter(1))[0]
            accumulative_prob *= node.accumulative_prob[previous_node]

            node = previous_node
            result.appendleft(previous_node.value)

        return list(result), accumulative_prob

    def __update_prob_recursive(self, start: GraphNode, probs: Ngram, previous_words: deque):
        fixed_previous_words = previous_words
        for next_node in start.next:
            previous_words = fixed_previous_words.copy()

            next_node: GraphNode = next_node  # just for type declaration

            # prior probability
            prior_prob = probs.probability(" ".join(previous_words))

            # union probability
            words = list(previous_words.copy())
            words.append(next_node.value)
            union_prob = probs.probability(" ".join(words))

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
