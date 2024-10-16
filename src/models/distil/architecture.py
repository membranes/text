"""Module architecture.py"""
import logging
import os

import transformers

import src.elements.arguments as ag
import src.elements.structures as sr
import src.functions.directories
import src.models.args
import src.models.distil.intelligence
import src.models.distil.metrics
import src.models.distil.settings


class Architecture:
    """
    Architecture
    """

    def __init__(self, arguments: ag.Arguments, enumerator: dict, archetype: dict):
        """

        :param arguments:
        :param enumerator:
        :param archetype:
        """

        self.__arguments = arguments
        self.__enumerator = enumerator
        self.__archetype = archetype

        # Directory preparation
        src.functions.directories.Directories().cleanup(path=self.__arguments.model_output_directory)

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
        args = src.models.args.Args(arguments=self.__arguments).exc()

        # Collator, Model, ETC.
        intelligence = src.models.distil.intelligence.Intelligence(enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.distil.metrics.Metrics(archetype=self.__archetype)

        # Settings
        settings = src.models.distil.settings.Settings(arguments=self.__arguments)

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
            n_trials=self.__arguments.N_TRIALS,
            direction='minimize',
            backend='ray',

            # scaling configuration
            resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},

            # tune configuration
            search_alg=settings.algorithm(),
            scheduler=settings.scheduler(), reuse_actors=True,

            # check point configuration
            # keep_checkpoints_num=8, checkpoint_score_attr='training_iteration',

            # run configuration: local_dir -> storage_path
            name='default', storage_path=os.path.join(self.__arguments.model_output_directory, 'ray'),
            verbose=0, progress_reporter=settings.reporting, log_to_file=True
        )

        return best
