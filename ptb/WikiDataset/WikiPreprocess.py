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
    _, word_dict = process_training_files(training_files, result_path)

    testing_files = get_data_list(os.path.join(fpath_list, 'testing'))
    process_testing_files(testing_files, result_path, word_dict=word_dict)


def process_data(file_paths: list, word_dict=None) -> list:
    # file_paths = file_paths[:1]# TODO for debug
    corpus: str = " ".join(list(map(lambda path: read_file_lines(path), file_paths)))
    corpus: str = process_corpus(corpus)
    corpus: list = corpus.splitlines(keepends=False)
    corpus = map(lambda s: process_line(s, word_dict=word_dict), corpus)
    corpus: list = list(filter(lambda s: len(s) > 1, corpus))  # remove single character or empty entries
    shuffle(corpus)

    return corpus


def process_testing_files(file_paths: list, data_dir: str, word_dict=None):
    lines = process_data(file_paths, word_dict=word_dict)

    split = split_dataset(lines, [1, 1])

    save_data(split[0], data_dir, 'test')
    save_data(split[1], data_dir, 'val')


def process_training_files(file_paths: list, data_dir: str):
    lines = process_data(file_paths)

    word_dict = build_and_save_dict(lines, data_dir)

    save_data(lines, data_dir, 'train')

    return lines, word_dict


def save_data(corpus: list, data_dir: str, file_type: str) -> None:
    with open(os.path.join(data_dir, file_type), 'a', encoding='utf8') as file:
        for line in corpus:
            file.write(f'{line}\n')


def build_and_save_dict(corpus: list, save_dir: str):
    word_set: list = list(set(" ".join(corpus).split()))
    word_set_file: str = " ".join(word_set)
    with open(os.path.join(save_dir, 'train.dict'), 'a', encoding='utf8') as file:
        file.write(word_set_file)

    return word_set


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


def split_dataset(corpus: list, ratio: list) -> list:
    """split a corpus to multiple parts"""
    idx = [int(float(i) / sum(ratio) * len(corpus)) for i in ratio]  # normalization

    split = []
    accumulation = 0
    for count in idx:
        split.append(corpus[accumulation:accumulation + count])
        accumulation += count

    return split


def reduce_continuous(s: str, reduce_num=False, reduce_letter=False, word_dict=None) -> str:
    """
    remove continuous symbols such as whitespaces, identities or digits

    """
    if reduce_num:
        s = re.sub(r'(<num>)+', '<num>', s)  # remove continuous num
    if reduce_letter:
        s = re.sub(r'(<letters>)+', '<letters>', s)  # remove continuous num

    if word_dict is not None:
        replacement = {k: "<unk>" for k in word_dict}
        re.sub('({})'.format('|'.join(map(re.escape, replacement.keys()))), lambda m: replacement[m.group()], s)

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


def process_line(line: str, word_dict=None) -> str:
    # replace continuous whitespaces to single one
    line = reduce_continuous(line, reduce_letter=False, reduce_num=False, word_dict=word_dict)

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
