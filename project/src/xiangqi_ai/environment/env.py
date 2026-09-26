"""Stateful environment shared by all agents."""


class StepResult:
    def __init__(self, state, reward, terminated, result):
        pass

class XiangqiEnv:
    def __init__(self, rules=None):
        pass

    def reset(self, state=None):
        pass

    def step(self, action):
        pass

    def get_legal_actions(self):
        pass

    def is_terminal(self):
        pass

    def get_winner(self):
        pass

    def clone_state(self):
        pass

    def render(self):
        pass
