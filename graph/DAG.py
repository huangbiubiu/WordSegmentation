import util
from graph.GraphNode import GraphNode
from util import Constant


class DAG:

    def __init__(self):
        self.start: GraphNode = GraphNode(Constant.START_SYMBOL)
        self.end: GraphNode = GraphNode(Constant.END_SYMBOL)

    def add_nodes(self, node, sub_sentence: str, word_dict: set):
        if sub_sentence == Constant.END_SYMBOL:
            node.next = self.end
            return
        for i in range(1, len(sub_sentence)):
            word: str = sub_sentence[:i]
            if word in word_dict:
                new_node = GraphNode(word)
                node.add_next(new_node)
                self.add_nodes(new_node, sub_sentence[i:], word_dict)
        pass

    @staticmethod
    def get_graph(sentence, word_dict):
        sentence = f"{sentence} {Constant.END_SYMBOL}"

        g = DAG()
        g.add_nodes(g.start, sentence, word_dict)

        return g


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    word_dict = util.construct_dict(dict_path)
    g = DAG.get_graph("去北京大学玩", word_dict)
    pass


if __name__ == '__main__':
    main()
