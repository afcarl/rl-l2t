__author__ = 'allentran'

import models

class DeepDPGTrainer(object):

    def __init__(self):

        self.model = models.DeepDPGModel()
        self.data = None

        self.replay_cache = None

    def train(self, epochs=100):
        pass
