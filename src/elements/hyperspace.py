import typing


class Hyperspace(typing.NamedTuple):

    learning_rate_distribution: list[float]
    weight_decay_distribution: list[float]
    weight_decay_choice: list[float]
    per_device_train_batch_size: list[int]
