''' Implements data structures needed for estimators '''

# The use of deque is temporary
from collections import deque

class Queue(object):
    def __init__(self, *args):
        self.deque = deque(*args)

    def push(self, x):
        self.deque.appendleft(x)
        return self

    def pop(self):
        return self.deque.pop()

    def peek(self):
        return self.deque[-1]


class Stack(object):
    def __init__(self, *args):
        self.deque = deque(*args)

    def push(self, x):
        self.deque.append(x)
        return self

    def pop(self):
        return self.deque.pop()

    def peek(self):
        return self.deque[-1]
