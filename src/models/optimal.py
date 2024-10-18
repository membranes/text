"""Module optimal.py"""
import os

import transformers

import src.elements.arguments as ag
import src.elements.structures as sr
import src.functions.directories
import src.models.algorithm
import src.models.trainee
import src.models.tuning
import src.models.metrics


class Optimal:
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

        # Intelligence
        self.__algorithm = src.models.algorithm.Algorithm(architecture=self.__arguments.name)

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

        # Training Arguments
        args = src.models.trainee.Trainee(arguments=self.__arguments).exc()

        # Model
        intelligence = self.__algorithm.exc(
            arguments=self.__arguments, enumerator=self.__enumerator, archetype=self.__archetype)

        # Metrics
        metrics = src.models.metrics.Metrics(archetype=self.__archetype)

        # Tuning
        tuning = src.models.tuning.Tuning(arguments=self.__arguments)

        # Temporary
        data_collator: transformers.DataCollatorForTokenClassification = (
            transformers.DataCollatorForTokenClassification(tokenizer=tokenizer))

        # Hence
        trainer = transformers.Trainer(
            model_init=intelligence.model,
            args=args, data_collator=data_collator,
            train_dataset=training.dataset, eval_dataset=validating.dataset,
            tokenizer=tokenizer,
            compute_metrics=metrics.exc
        )

        best = trainer.hyperparameter_search(
            hp_space=tuning.hp_space,
            compute_objective=tuning.compute_objective,
            n_trials=self.__arguments.N_TRIALS,
            direction='minimize',
            backend='ray',

            # scaling configuration
            resources_per_trial={'cpu': self.__arguments.N_CPU, 'gpu': self.__arguments.N_GPU},

            # tune configuration
            search_alg=tuning.algorithm(),
            scheduler=tuning.scheduler(), reuse_actors=True,

            # check point configuration
            # keep_checkpoints_num=8, checkpoint_score_attr='training_iteration',

            # run configuration: local_dir -> storage_path
            name='default', storage_path=os.path.join(self.__arguments.model_output_directory, 'ray'),
            verbose=1, progress_reporter=tuning.reporting, log_to_file=True
        )

        return best