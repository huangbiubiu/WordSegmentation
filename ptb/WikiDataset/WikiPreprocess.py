import os
import re
from random import shuffle

import zhon.hanzi
import sys
import string


def main(fpath_list: list, result_path: str, min_len=7, limit_line_cnt=None) -> None:
    """
    :param fpath_list: file path list of dataset
    :param result_path: the target dir path to save dataset
    :param min_len: minimum length of a output entry
    :param limit_line_cnt: only read limit_line_cnt lines from input files. For test purpose.

    Note: this method load all data into the memory for simplicity. ~20 Gigabytes memory is required
    for processing all data. Make sure set limit_line_cnt when testing, and the memory is enough when
    process the whole dataset.
    """
    corpus = " ".join(list(map(lambda path: read_file_lines(path, limit_line_cnt), fpath_list)))
    corpus = preprocess(corpus)
    corpus = remove_multiple_whitespace(corpus)
    corpus = split_words(corpus).splitlines(keepends=False)
    corpus = list(filter(lambda s: str_len(s) > min_len, corpus))
    shuffle(corpus)

    for fname, data_set in zip(["train", "test", "val"], split_dataset(corpus)):
        with open(os.path.join(result_path, fname), 'w', encoding='utf8') as file:
            file.write("\n".join(data_set))
        pass


def read_file_lines(file_path: str, limit=None) -> str:
    with open(file_path, encoding='utf8') as file:
        if limit is None:
            lines = " ".join(file.readlines()[:limit])
        else:
            line_cnt = 0
            lines = []
            for line in file:
                lines.append(line)
                line_cnt += 1
                if line_cnt >= limit:
                    break
            lines = "".join(lines)
        return lines


def split_dataset(corpus: list, ratio=None) -> (str, str, str):
    if ratio is None:
        ratio = [0.85, 0.075]
    total_len = len(corpus)
    training_size = int(total_len * ratio[0])
    testing_size = int(total_len * ratio[1])

    return corpus[: training_size], corpus[training_size: training_size + testing_size], corpus[
        training_size + testing_size:]


def remove_multiple_whitespace(s: str) -> str:
    s = re.sub(r'(<num>)+', '<num>', s)  # remove continuous num
    s = re.sub(r'(<letters>)+', '<letters>', s)  # remove continuous num
    return re.sub(r'\s+', '\n', s)


def str_len(s: str) -> int:
    return len(re.sub(r'(<.+>)', 'I', s))


def split_words(s: str) -> str:
    s = re.sub(r'([\u4e00-\u9fa5])|(<.+>)', r'\1 ', s)
    return s


def preprocess(s: str) -> str:
    # change to lower letter
    s = s.lower()

    # remove punctuation
    punctuations = zhon.hanzi.punctuation + string.punctuation
    table = str.maketrans({**{k: "\n" for k in punctuations},
                           **{str(k): "<num>" for k in range(10)},
                           **{k: "<letters>" for k in string.ascii_lowercase}})

    return s.translate(table)

    pass


if __name__ == '__main__':
    file_path_list = sys.argv[1:]
    main(file_path_list, '../../data', limit_line_cnt=None)
