"""Module settings"""
import logging
import ray
import ray.tune
import ray.tune.schedulers as rts
import ray.tune.search.optuna as opt

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

        # Space
        self.__space = {'learning_rate': ray.tune.uniform(lower=0.000016, upper=0.000020),
                        'weight_decay': ray.tune.uniform(lower=0.0, upper=0.00001)}

    @staticmethod
    def compute_objective(metric):
        """

        :param metric: A placeholder
        :return:
        """

        return metric['eval_loss']

    def hp_space(self, trial):
        """

        :param trial: A placeholder
        :return:
        """

        logging.info(trial)

        return self.__space

    def scheduler(self):
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html

        Search algorithms cannot be used with PopulationBasedTraining schedulers.

        return rts.PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_loss', mode='min',
            perturbation_interval=self.__perturbation_interval,
            hyperparam_mutations={
                'learning_rate': ray.tune.uniform(lower=0.000016, upper=0.000020),
                'weight_decay': ray.tune.uniform(lower=0.0, upper=0.00001)
            },
            quantile_fraction=0.25,
            resample_probability=0.25)

        :return:
        """

        return rts.ASHAScheduler(
            time_attr='training_iteration', metric='eval_loss', mode='min')



    @staticmethod
    def algorithm():
        """

        :return:
        """

        return opt.OptunaSearch(metric='eval_loss', mode='min')

    @staticmethod
    def reporting():
        """
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html

        :return:
        """

        return ray.tune.CLIReporter(
            parameter_columns=['learning_rate', 'weight_decay', 'per_device_training_batch_size'],
            metric_columns=['eval_loss', 'precision', 'recall', 'f1'])
