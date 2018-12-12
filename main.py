import zhon.hanzi
import string
from graph.ProbCalculator import ProbCalculator

CORPUS_PATH = "./data/train"
DICT_PATH = "./data/train.dict"

PUNCTUATIONS: set = set(zhon.hanzi.punctuation + string.punctuation + " ")


def __find_next_punctuation(sentence: str, start=0) -> int:
    if start >= len(sentence):
        raise IndexError(f"start index {start} out of range")

    for i in range(start, len(sentence)):
        if sentence[i] in PUNCTUATIONS:
            return i
    return -1


def __cut(sentence: str, graph: ProbCalculator) -> str:
    if len(sentence) == 0:
        return ""

    next_punctuation = 0
    segments = []
    while next_punctuation != -1:
        if len(segments) != 0:
            start = next_punctuation + 1
            next_punctuation = __find_next_punctuation(sentence, start=next_punctuation + 1)
        else:
            start = next_punctuation
            next_punctuation = __find_next_punctuation(sentence, start=next_punctuation)

        seg = graph.calc(sentence[start:next_punctuation])
        segments.append(seg)
        segments.append(sentence[next_punctuation:next_punctuation + 1])  # add punctuation
        pass

    return " ".join(segments)


def main(data_path: str):
    graph = ProbCalculator(dict_path=DICT_PATH,
                           corpus_path=CORPUS_PATH,
                           ngram_size=2)
    with open(data_path, encoding='utf8') as file:
        lines = file.readlines()
    for line in lines:
        print(__cut(line, graph))
    pass


if __name__ == '__main__':
    main(data_path="./data/test.txt")
