''' Implements data structures needed for models '''

from libc.stdlib cimport malloc

cdef struct QueueNode:
    float value
    QueueNode* back
    QueueNode* front

cdef class Queue:

    cdef QueueNode* top
    cdef QueueNode* bot
    cdef QueueNode* node
    cdef float value

    def __cinit__(self):
        self.bot = <QueueNode*>malloc(sizeof(QueueNode))
        self.bot.value = -1
        self.bot.front = self.top

        self.top = <QueueNode*>malloc(sizeof(QueueNode))
        self.top.value = -1
        self.top.back = self.bot

    def push(self, float x):
        if self.top.value == -1:
            self.top.value = x
        elif self.bot.value == -1:
            self.bot.value = x
        else:
            node = <QueueNode*>malloc(sizeof(QueueNode))
            node.value = x
            node.front = self.bot
            self.bot.back = node
            self.bot = node
        return self

    def pop(self):
        if self.top.value == -1:
            raise IndexError('The queue is empty.')
        elif self.bot.value == -1:
            value = self.top.value
            self.top.value = -1
            return value
        else:
            value = self.top.value
            if self.top.back == self.bot:
                self.top.value = self.bot.value
                self.bot.value = -1
            else:
                self.top = self.top.back
            return value

    def peek(self):
        return self.top.value

cdef struct StackNode:
    float value
    StackNode* back

cdef class Stack:
    
    cdef StackNode* top
    cdef float value

    def __cinit__(self):
        self.top = <StackNode*>malloc(sizeof(StackNode))
        self.top.value = -1

    def push(self, float x):
        node = <StackNode*>malloc(sizeof(StackNode))
        node.value = x
        node.back = self.top
        self.top = node
        return self

    def pop(self):
        if self.top.value == -1:
            raise IndexError('The stack is empty.')
        else:
            value = self.top.value
            self.top = self.top.back
            return value

    def peek(self):
        return self.top.value
