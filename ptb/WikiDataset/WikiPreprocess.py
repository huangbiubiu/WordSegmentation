import os
import re
from random import shuffle

import zhon.hanzi
import sys
import string

import unicodedata


def main(fpath_list: str, result_path: str, min_len=7, limit_line_cnt=None) -> None:
    """
    :param fpath_list: file path list of dataset
    :param result_path: the target dir path to save dataset
    :param min_len: minimum length of a output entry
    :param limit_line_cnt: only read limit_line_cnt lines from input files. For test purpose.

    Note: this method load all data into the memory for simplicity. ~20 Gigabytes memory is required
    for processing all data. Make sure set limit_line_cnt when testing, and the memory is enough when
    process the whole dataset.
    """
    training_files = get_data_list(os.path.join(fpath_list, 'training'))
    process_training_files(training_files, result_path)


def process_training_files(file_paths: list, data_dir: str):
    # corpus: str = " ".join(list(map(lambda path: read_file_lines(path), file_paths)))
    file_paths = file_paths[:1]
    corpus: str = " ".join(list(map(lambda path: read_file_lines(path), file_paths)))  # TODO for debug
    corpus: str = process_corpus(corpus)
    corpus: list = corpus.splitlines(keepends=False)
    corpus = map(process_line, corpus)
    corpus: list = list(filter(lambda s: len(s) > 1, corpus))  # remove single character or empty entries
    shuffle(corpus)

    build_dict(corpus, data_dir)


def build_dict(corpus: list, save_dir: str):
    word_set: list = list(set(" ".join(corpus).split()))
    word_set_file: str = " ".join(word_set)
    with open(os.path.join(save_dir, 'train.dict'), 'a', encoding='utf8') as file:
        file.write(word_set_file)


def get_data_list(data_dir: str):
    data_list = []
    for file in os.listdir(data_dir):
        if file.endswith(".utf8"):
            data_list.append(os.path.join(data_dir, file))

    return data_list


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


def reduce_continuous(s: str, reduce_num=False, reduce_letter=False) -> str:
    """
    remove continuous symbols such as whitespaces, identities or digits

    """
    if reduce_num:
        s = re.sub(r'(<num>)+', '<num>', s)  # remove continuous num
    if reduce_letter:
        s = re.sub(r'(<letters>)+', '<letters>', s)  # remove continuous num
    s = ' '.join(s.split())  # remove continuous whitespaces
    return s


def str_len(s: str) -> int:
    return len(re.sub(r'(<.+>)', 'I', s))


def split_words(s: str) -> str:
    """
    Add whitespace after each characters and identities.

    Only use when train a character-based LM. Not use when the corpus is already split.
    :param s: corpus
    :return: processed corpus
    """
    s = re.sub(r'([\u4e00-\u9fa5])|(<.+>)', r'\1 ', s)
    return s


def process_line(line: str) -> str:
    # replace continuous whitespaces to single one
    line = reduce_continuous(line, reduce_letter=False, reduce_num=False)

    return line.strip()
    pass


def process_corpus(corpus: str) -> str:
    """
    replace special characters
    """
    # replace full-width characters with half-width characters
    corpus = unicodedata.normalize('NFKC', corpus)

    # change to lower letter
    corpus = corpus.lower()

    # remove special chars
    table = str.maketrans({**{k: "\n" for k in zhon.hanzi.stops},
                           **{k: " " for k in zhon.hanzi.non_stops + string.punctuation},
                           **{str(k): "N" for k in range(10)},
                           **{k: "N" for k in string.ascii_lowercase}})
    corpus = corpus.translate(table)

    return corpus

    pass


if __name__ == '__main__':
    input_path = sys.argv[1]
    # output_path = sys.argv[2]
    main(input_path, '../../data', limit_line_cnt=None)
