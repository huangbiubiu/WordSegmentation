class GraphNode:
    start: int
    end: int
    next: list
    previous: list
    prior_prob: float
    union_prob: float
    value: str

    accumulative_prob: dict

    def __init__(self, word: str, start: int, end: int):
        self.value: str = word

        self.accumulative_prob = {}

        self.previous: list = []
        self.next: list = []

        self.start = start
        self.end = end

    def add_next(self, next_node):
        self.next.append(next_node)
        next_node.previous.append(self)

    def __str__(self):
        return self.value
