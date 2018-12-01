class GraphNode:
    def __init__(self, word: str):
        self.value: str = word

        self.upstream_prob: float
        self.self_prob: float

        self.previous: list = []
        self.next: list = []

    def add_next(self, next_node):
        self.next.append(next_node)
        next_node.previous.append(self)

    def __str__(self):
        return self.value
