from __future__ import division

import numpy as np

#-----------------------------------------------------------------------------
# A general purpose Distribution class for finite discrete random variables
#

class Distribution(dict):
    """
    The Distribution class extends the Python dictionary such that
    each key's value should correspond to the probability of the key.

    For example, here's how you can create a random variable X that takes on
    value 'spam' with probability .7 and 'eggs' with probability .3:

    X = Distribution()
    X['spam'] = .7
    X['eggs'] = .3

    Methods
    -------
    normalize():
      scales all the probabilities so that they sum to 1
    get_mode():
      returns an item with the highest probability, breaking ties arbitrarily
    sample():
      draws a sample from the Distribution
    """
    def __missing__(self, key):
        # if the key is missing, return probability 0
        return 0

    def normalize(self):
        normalization_constant = sum(self.values())
        if normalization_constant == 0:
            return
        for key in self.keys():
            self[key] /= normalization_constant

    def get_mode(self):
        maximum = -1
        arg_max = None

        for key in self.keys():
            if self[key] > maximum:
                arg_max = key
                maximum = self[key]

        return arg_max

    def sample(self):
        self.normalize()
        keys  = []
        probs = []
        for key, prob in self.items():
            keys.append(key)
            probs.append(prob)
        if not keys:
            return None
        rand_idx = np.where(np.random.multinomial(1, probs))[0][0]
        return keys[rand_idx]
