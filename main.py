from graph.ProbCalculator import ProbCalculator
import jieba
CORPUS_PATH = "./data/train"
DICT_PATH = "./data/train.dict"


def main(data_path: str):
    graph = ProbCalculator(dict_path=DICT_PATH,
                           corpus_path=CORPUS_PATH,
                           ngram_size=2)

    with open(data_path, encoding='utf8') as file:
        lines = file.readlines()
    for line in lines:
        print(jieba.cut(line))
    pass


if __name__ == '__main__':
    main(data_path="./data/test.txt")
