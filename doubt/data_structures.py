''' Implements data structures needed for models '''

from collections import deque

class Stack:
    ''' Stack implementation, being a LIFO data structure.

    >>> stack = Stack()
    >>> stack.push(2)
    >>> stack.push(5)
    >>> stack.pop()
    5
    >>> stack.peek()
    2
    >>> stack.pop()
    2
    >>> stack = Stack(['foo', 'bar'])
    >>> stack.pop()
    'bar'
    '''
    def __init__(self, iterable = []):
        self.deque = deque(iterable)

    def push(self, value):
        return self.deque.append(value)

    def pop(self):
        return self.deque.pop()

    def peek(self):
        return list(self.deque)[-1]

class Queue:
    ''' Queue implementation, being a FIFO data structure.

    >>> queue = Queue()
    >>> queue.push(2)
    >>> queue.push(5)
    >>> queue.pop()
    2
    >>> queue.peek()
    5
    >>> queue.pop()
    5
    >>> queue = Queue(['foo', 'bar'])
    >>> queue.pop()
    'bar'
    '''
    def __init__(self, iterable = []):
        self.deque = deque(iterable)

    def push(self, value):
        return self.deque.appendleft(value)

    def pop(self):
        return self.deque.pop()

    def peek(self):
        return list(self.deque)[-1]
