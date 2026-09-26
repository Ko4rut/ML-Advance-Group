"""Model versions, metadata and training state."""


class CheckpointStore:
    def __init__(self, directory):
        pass

    def save(self, network, optimizer, metadata):
        pass

    def load(self, path):
        pass

    def list_versions(self):
        pass
