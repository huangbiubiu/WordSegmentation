def get_graph(sentence, word_dict: set):
    graph = {}
    for i in range(len(sentence)):
        graph[i] = [i]
        for j in range(i + 1, len(sentence)):
            word = sentence[i:j + 1]
            if word in word_dict:
                graph[i].append(j)
    return graph


def construct_dict(path: str) -> set:
    with open(path, encoding='utf8') as f:
        return set(f.readline().split())


def main():
    dict_path: str = "/home/hyh/projects/CLProject/WordSegmentation/data/train.dict"
    word_dict = construct_dict(dict_path)
    g = get_graph("去北京大学玩", word_dict)
    pass


if __name__ == '__main__':
    main()
