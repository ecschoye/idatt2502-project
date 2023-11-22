from collections import deque


class CustomDeque:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.deque = deque(maxlen=max_size)

    def add_to_front(self, item):
        if len(self.deque) == self.max_size:
            self.deque.pop()
        self.deque.appendleft(item)

    def get_elements(self):
        return list(self.deque)

    def get_sum(self):
        return sum(self.deque)

    def get_length(self):
        return len(self.deque)
