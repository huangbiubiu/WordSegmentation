class GraphNode:
    next: list
    previous: list
    prior_prob: float
    union_prob: float
    value: str

    def __init__(self, word: str):
        self.value: str = word

        self.accumulative_prob: dict = {}

        self.previous: list = []
        self.next: list = []

    def add_next(self, next_node):
        self.next.append(next_node)
        next_node.previous.append(self)

    def __str__(self):
        return self.value
