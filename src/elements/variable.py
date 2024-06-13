import typing

class Variable(typing.NamedTuple):
    """

    Attributes
    ----------

    MAX_LENGTH : int
        The maximum number of tokens

    TRAIN_BATCH_SIZE : int
        The training stage data batch size

    VALID_BATCH_SIZE : int
        The validation stage data batch size

    EPOCHS : int
        The number of epochs

    LEARNING_RATE : float
        The learning rate

    MAX_GRADIENT_NORM : int
        The maximum gradient norm
    """

    MAX_LENGTH: int = 128
    TRAIN_BATCH_SIZE: int = 4
    VALID_BATCH_SIZE: int = 2
    EPOCHS: int = 1
    LEARNING_RATE: float = 1e-05
    MAX_GRADIENT_NORM: int = 10
