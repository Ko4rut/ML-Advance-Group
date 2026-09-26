"""Agent skeleton; implementation is deferred."""

from .base import Agent


class DQNAgent(Agent):
    def __init__(self, network, observation_encoder, action_encoder, epsilon=0.0):
        pass

    def select_action(self, state, budget=None):
        pass
