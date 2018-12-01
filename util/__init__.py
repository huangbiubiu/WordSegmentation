class Constant:
    START_SYMBOL = 'S'
    END_SYMBOL = 'E'


def construct_dict(path: str) -> set:
    with open(path, encoding='utf8') as f:
        word_dict = set(f.readline().split())
        word_dict.add(Constant.START_SYMBOL)
        word_dict.add(Constant.END_SYMBOL)
        return word_dict
