"""Common agent interface and decision metadata."""


class SearchBudget:
    def __init__(self, max_depth=None, max_nodes=None, max_simulations=None, time_limit=None):
        pass

class AgentDecision:
    def __init__(self, move, nodes=0, simulations=0, policy=None):
        pass

class Agent:
    def select_action(self, state, budget=None):
        pass

    def reset(self):
        pass
