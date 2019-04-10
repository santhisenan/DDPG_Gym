from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque()

    def add_to_memory(self, experience):
        if(self._count <= self._buffer_size):
            self._buffer.append(experience)
            self._count += 1
        else:
            self._buffer.popleft()
            self._buffer.append(experience)

    def size(self):
        return self._count

    def sample_from_memory(self, batch_size=32):
        '''
        If the number of elements in the replay memory is less than the required 
        batch_size, then return only those elements present in the memory, else
        return 'batch_size' number of elements.
        '''

        _available_batch_length = \
            self._count if self._count < batch_size else batch_size

        batch = random.sample(self._buffer, _available_batch_length)
        return batch

    def clear(self):
        self._buffer.clear()
        self._count = 0
