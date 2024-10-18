import typing


class Hyperspace(typing.NamedTuple):

    learning_rate_lower: float
    learning_rate_upper: float
    weight_decay_lower: float
    weight_decay_upper: float
    weight_decay: list[float]
    per_device_train_batch_size: list[int]
