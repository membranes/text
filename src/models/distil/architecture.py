"""Module architecture.py"""
import logging
import os

import transformers

import src.elements.structures as sr
import src.elements.variable as vr
import src.models.distil.arguments
import src.models.distil.intelligence
import src.models.distil.metrics
import src.models.distil.parameters as pr
import src.models.distil.settings
import src.models.distil.storage


class Architecture:
    """
    Architecture
    """

    def __init__(self, variable: vr.Variable, enumerator: dict, archetype: dict):
        """

        :param variable:
        :param enumerator:
        :param archetype:
        """

        self.__variable = variable
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Parameters
        self.__parameters = pr.Parameters()

        # Directory preparation
        src.models.distil.storage.Storage().exc(path=self.__parameters.storage_path)

    def __call__(self, training: sr.Structures, validating: sr.Structures,
                 tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase) -> transformers.trainer_utils.BestRun:
        """
        https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/trainer#transformers.Trainer
        https://docs.ray.io/en/latest/tune/api/doc/ray.tune.run.html

        :param training:
        :param validating:
        :param tokenizer:
        :return:
        """

        # Arguments
        args = src.models.distil.arguments.Arguments(variable=self.__variable).exc()

        # Collator
        intelligence = src.models.distil.intelligence.Intelligence(enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.distil.metrics.Metrics(archetype=self.__archetype)

        # Settings
        settings = src.models.distil.settings.Settings(variable=self.__variable)

        # Hence
        trainer = transformers.Trainer(
            model_init=intelligence.model,
            args=args, data_collator=intelligence.collator(tokenizer),
            train_dataset=training.dataset, eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc
        )

        best = trainer.hyperparameter_search(
            hp_space=settings.hp_space,
            compute_objective=settings.compute_objective,
            n_trials=self.__variable.N_TRIALS,
            direction='minimize',
            backend='ray',

            # scaling configuration
            resources_per_trial={'cpu': self.__variable.N_CPU, 'gpu': self.__variable.N_GPU},

            # tune configuration
            search_alg=settings.algorithm(),
            scheduler=settings.scheduler(), reuse_actors=True,

            # check point configuration
            # keep_checkpoints_num=8, checkpoint_score_attr='training_iteration',

            # run configuration: local_dir -> storage_path
            name='default', storage_path=os.path.join(self.__parameters.storage_path, 'ray'),
            verbose=0, progress_reporter=settings.reporting, log_to_file=True
        )

        return best
