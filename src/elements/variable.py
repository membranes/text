import typing

class Variable(typing.NamedTuple):

    MAX_LENGTH: int = 128
    TRAIN_BATCH_SIZE: int = 4
    VALID_BATCH_SIZE: int = 2
    EPOCHS: int = 1
    LEARNING_RATE: float = 1e-05
    MAX_GRADIENT_NORM: int = 10
