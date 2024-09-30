"""Module settings"""

import ray
import ray.tune
import ray.tune.schedulers as rts

import src.elements.variable as vr


class Settings:
    """
    Class Settings
    """

    def __init__(self, variable: vr.Variable):
        """

        :param variable: A suite of values for machine learning
                         model development
        """

        self.__variable = variable

        # Re-visit
        self.__perturbation_interval = 2

    @staticmethod
    def hp_space(trial):
        """
        If the search algorithm depends on a continuous distribution, e.g., BayesOpt,
        do not use a categorical sample space, e.g., ray.tune.choice

        'per_device_train_batch_size': ray.tune.choice([4, 16, 32]),
        'num_train_epochs': ray.tune.choice([2, 4])

        :return:
        """

        return {'learning_rate': ray.tune.quniform(lower=0.001, upper=0.002, q=0.001),
                'weight_decay': ray.tune.quniform(lower=0.01, upper=0.02, q=0.01),
                'per_device_train_batch_size': ray.tune.choice([4, 16])}

    @staticmethod
    def compute_objective(metric):
        """

        :param metric:
        :return:
        """

        return metric['eval_loss']

    def scheduler(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html

        Leads on from hp_space

        :return:
        """

        return rts.PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_loss', mode='min',
            perturbation_interval=self.__perturbation_interval,
            hyperparam_mutations={
                'learning_rate': ray.tune.uniform(lower=0.001, upper=0.1),
                'weight_decay': ray.tune.uniform(lower=0.01, upper=0.1)
            },
            quantile_fraction=0.25,
            resample_probability=0.25
        )

    @staticmethod
    def reporting():
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html

        :return:
        """

        return ray.tune.CLIReporter(
            parameter_columns=['learning_rate', 'weight_decay', 'per_device_training_batch_size', 'per_device_eval_batch_size'],
            metric_columns=['eval_loss', 'precision', 'recall', 'f1'])
