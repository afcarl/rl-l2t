__author__ = 'allentran'

import theano
from theano_layers import layers
import numpy as np
import theano.tensor as TT

class DeepDPGModel(object):

    def __init__(self, n_minibatch, n_assets, n_actions, k_info, preprocessed_size, lstm_size, merge_size):

        self.policy_model = DeepPolicyNetwork(
            n_minibatch,
            n_assets,
            n_actions,
            k_info,
            preprocessed_size,
            lstm_size,
            merge_size
        )
        self.q_model = DeepQNetwork

# need current weights and target weights
# update weights (current via SGD, target from post SGD)
# need to generate actions (buy/sell per asset, buys as fraction of cash, sell as fraction of asset) - maybe just sigmoid

# prices are T x minibatch x N (make sure these are scaled)
# info is T x minibatch x k
# time to trade is T x minibatch x N

# deterministic states are 1 x minibatch x s
# holdings x N + 1

class DeepPolicyNetwork(object):

    def __init__(self, n_minibatch, n_assets, n_actions, k_info, preprocessed_size, lstm_size, merge_size, seed=1692):

        np.random.seed(seed)
        self.srng = TT.shared_randomstreams.RandomStreams(seed)

        self.n_minibatch = n_minibatch
        self.n_actions = n_actions

        self.layers = []

        # non deterministic states
        self.prices = TT.tensor3()
        self.info = TT.tensor3()
        self.next_trade = TT.tensor3()

        self.det_states = TT.matrix()

        # deterministic states

        states = TT.concatenate(
            [
                self.prices,
                self.info,
                self.next_trade
            ],
            axis=2
        )

        nondet_prepreprocessor = layers.DenseLayer(
            states,
            n_assets * 2 + k_info,
            preprocessed_size,
            TT.nnet.relu,
            normalize_axis=1
        )

        nondet_processor = layers.LSTMLayer(
            nondet_prepreprocessor.h_outputs,
            lstm_size,
            preprocessed_size,
            normalize_axis=1
        )

        det_processor = layers.DenseLayer(
            self.det_states,
            n_assets * 2 + 1,
            preprocessed_size,
            TT.nnet.relu,
            normalize_axis=1
        )

        merger = layers.DenseLayer(
            TT.concatenate(
                [
                    nondet_processor.h_outputs[-1, :, :],
                    det_processor.h_outputs
                ],
                axis=1
            ),
            preprocessed_size * 2,
            merge_size,
            TT.nnet.relu,
            normalize_axis=0
        )

        merger2 = layers.DenseLayer(
            merger.h_outputs,
            merge_size,
            merge_size,
            TT.nnet.relu,
            normalize_axis=0
        )

        self.action_layer = layers.DenseLayer(
            merger2.h_outputs,
            merge_size,
            n_actions,
            TT.nnet.softmax,
            normalize_axis=0
        )

        self.layers += [
            nondet_prepreprocessor,
            nondet_processor,
            det_processor,
            merger,
            merger2,
            self.action_layer
        ]

    def update_target_weights(self):
        pass

    def get_noisy_action(self, std):
        actions = self.action_layer.h_outputs
        return actions + self.srng.normal(size=actions.shape, std=std)

# need current weights and target weights
# update weights (current via SGD, target from post SGD)
# need gradients wrt to theta AND actions
class DeepQNetwork(object):

    def __init__(self, n_minibatch, n_assets, n_actions, k_info, preprocessed_size, lstm_size, merge_size, dense_sizes, seed=1692):

        self.prices = TT.tensor3()
        self.info = TT.tensor3()
        self.next_trade = TT.tensor3()

        self.actions = TT.matrix()
        self.det_states = TT.matrix()

        states = TT.concatenate(
            [
                self.prices,
                self.info,
                self.next_trade
            ],
            axis=2
        )

        nondet_prepreprocessor = layers.DenseLayer(
            states,
            n_assets * 2 + k_info,
            preprocessed_size,
            TT.nnet.relu,
            normalize_axis=1,
            dupe_weights=True
        )

        nondet_processor = layers.LSTMLayer(
            nondet_prepreprocessor.h_outputs,
            lstm_size,
            preprocessed_size,
            normalize_axis=1,
            dupe_weights=True
        )

        det_processor = layers.DenseLayer(
            self.det_states,
            n_assets * 2 + 1,
            preprocessed_size,
            TT.nnet.relu,
            normalize_axis=1,
            dupe_weights=True
        )

        state_merger = layers.DenseLayer(
            TT.concatenate(
                [
                    nondet_processor.h_outputs[-1, :, :],
                    det_processor.h_outputs
                ],
                axis=1
            ),
            preprocessed_size * 2,
            merge_size,
            TT.nnet.relu,
            normalize_axis=0,
            dupe_weights=True
        )

        state_and_actions = TT.concatenate(
            [
                state_merger.h_outputs,
                self.actions
            ],
            axis=1
        )

        dense_layer = layers.DenseLayer(
            state_and_actions,
            merge_size + n_actions,
            dense_sizes[0],
            activation=TT.nnet.relu,
            normalize_axis=0,
            dupe_weights=True
        )

        dense_layer2 = layers.DenseLayer(
            dense_layer.h_outputs,
            dense_sizes[0],
            dense_sizes[1],
            activation=TT.nnet.relu,
            normalize_axis=0,
            dupe_weights=True
        )

        output_layer = layers.DenseLayer(
            dense_layer2.h_outputs,
            dense_sizes[1],
            1,
            normalize_axis=0,
            dupe_weights=True
        )

        self.q_sa = output_layer.h_outputs

        self.layers = [
            nondet_prepreprocessor,
            nondet_processor,
            det_processor,
            state_merger,
            dense_layer,
            dense_layer2,
            output_layer
        ]

