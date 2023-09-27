import numpy as np


class Tree:
    def set_reward(self, r):
        pass

class Policies:
    def __init__(self, name):
        self._name = name
        self._tree = Tree()

    def get_output(self, x):
        return np.random.randint(21)

    def set_reward(self, r):
        pass
