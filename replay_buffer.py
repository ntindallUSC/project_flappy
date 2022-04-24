import collections
import random

class ReplayBuffer(object):
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)

    def append(self, state, action, reward, state_next, done):
        self.buffer.append((state, action, reward, state_next, done))

    def sample(self, n_sample):
        return random.sample(self.buffer, n_sample)

    def clear(self):
        self.buffer.clear()

    @property
    def size(self):
        return len(self.buffer)
    
