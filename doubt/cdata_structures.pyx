''' Implements data structures needed for models '''

from libc.stdlib cimport malloc

cdef struct QueueNode:
    void* value
    QueueNode* back
    QueueNode* front

cdef class Queue:
    #cdef QueueNode* top
    #cdef QueueNode* bot
    #cdef void* popped_value

    def __cinit__(self):
        self.bot = NULL
        self.top = NULL

    cdef public int push(self, void* x):
        cdef QueueNode* node = <QueueNode*>malloc(sizeof(QueueNode))
        node.value = x

        if self.top.value == NULL:
            self.top = node
            self.bot = node
        elif self.bot.value == NULL:
            self.bot = node
            node.front = self.top
        else:
            node.front = self.bot
            self.bot.back = node
            self.bot = node
        return 1

    cdef public void* pop(self):
        if self.top.value == NULL:
            raise IndexError('The queue is empty.')
        elif self.bot.value == NULL:
            popped_value = self.top.value
            self.top.value = NULL
            return popped_value
        else:
            popped_value = self.top.value
            if self.top.back == self.bot:
                self.top.value = self.bot.value
                self.bot.value = NULL
            else:
                self.top = self.top.back
            return popped_value

    cdef public void* peek(self):
        return self.top.value

cdef struct StackNode:
    void* value
    StackNode* back

cdef class Stack:
    #cdef StackNode* top
    #cdef void* popped_value

    def __cinit__(self):
        self.top = NULL

    cdef int push(self, void* x):
        cdef StackNode* node = <StackNode*>malloc(sizeof(StackNode))
        node.value = x
        node.back = self.top
        self.top = node
        return 1

    cdef void* pop(self):
        if self.top.value == NULL:
            raise IndexError('The stack is empty.')
        else:
            popped_value = self.top.value
            self.top = self.top.back
            return popped_value

    cdef void* peek(self):
        return self.top.value
