"""Versioned game records and replay validation."""


class GameRecord:
    def __init__(self, metadata, moves, result):
        pass

    def to_dict(self):
        pass

    @classmethod
    def from_dict(cls, data):
        pass

class GameRecordStore:
    def save(self, record, path):
        pass

    def load(self, path):
        pass

    def replay(self, record, engine):
        pass
