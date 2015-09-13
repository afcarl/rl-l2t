__author__ = 'allentran'

import models



class TestNetwork(object):

    def setUp(self):

        self.n_minibatch = 32
        self.n_assets = 2
        self.n_actions = 2 * self.n_assets
        self.k_info = 16
        self.preprocessed_size = 10
        self.lstm_size = 5
        self.merge_size = 5
        self.q_dense_sizes = [20, 10]

    def policy_test(self):

        policy_network = models.DeepPolicyNetwork(
            self.n_minibatch,
            self.n_assets,
            self.n_actions,
            self.k_info,
            self.preprocessed_size,
            self.lstm_size,
            self.merge_size
        )

        policy_network.get_noisy_action(1e-2)

    def q_test(self):

        q_network = models.DeepQNetwork(
            self.n_minibatch,
            self.n_assets,
            self.n_actions,
            self.k_info,
            self.preprocessed_size,
            self.lstm_size,
            self.merge_size,
            self.q_dense_sizes
        )

    def gen_fake_states(self):
        pass

