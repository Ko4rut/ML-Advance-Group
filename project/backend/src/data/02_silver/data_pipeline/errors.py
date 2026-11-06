"""Stable quarantine error types."""


class SilverValidationError(ValueError):
    def __init__(self, code: str, message: str, ply: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.ply = ply
