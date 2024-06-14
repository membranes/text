
import logging
import transformers
import transformers.modeling_outputs as tm
import sklearn.metrics as sm

import torch
import torch.utils.data as tu

import src.models.bert.parameters

import src.elements.variable



class Modelling:

    def __init__(self, variable: src.elements.variable.Variable,
                 enumerator: dict, dataloader: tu.DataLoader):
        """

        :param enumerator: The labels enumerator
        """

        self.__variable = variable
        self.__dataloader = dataloader

        # The device for computation
        self.__device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained
        # https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig
        self.__parameters = src.models.bert.parameters.Parameters()

        logging.info('\n\nPretrained Model\n')
        self.__model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        self.__model.to(self.__device)

    def __train(self):
        """

        :return:
        """

        # For measures & metrics
        loss_: float = 0
        steps_: int = 0
        accuracy_ = 0

        # For estimates
        __labels = []
        __predictions = []

        # Preparing a training epoch ...
        self.__model.train()
        logging.info(self.__model.__dict__)

        index: int
        batch: dict
        for index, batch in enumerate(self.__dataloader):

            inputs_: torch.Tensor = batch['input_ids'].to(self.__device, dtype = torch.long)
            labels_: torch.Tensor = batch['labels'].to(self.__device, dtype = torch.long)
            attention_mask_: torch.Tensor = batch['attention_mask'].to(self.__device, dtype = torch.long)

            # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput
            bucket: tm.TokenClassifierOutput = self.__model(input_ids=inputs_, attention_mask=attention_mask_, labels=labels_)

            logging.info(bucket.keys())
            loss_ += bucket.loss.item()
            steps_ += 1

            if (index % 100) == 0:
                print(f'Average epoch loss, after step {steps_}: {loss_/steps_}')

            # Accuracy
            targets = labels_.view(-1)
            active = labels_.view(-1).ne(100)
            __labels.extend(torch.masked_select(targets, active))

            # Predictions
            logits = bucket.logits.view(-1, self.__model.config.num_labels)
            __predictions.extend(torch.argmax(logits, dim=1))

            # sm.accuracy_score()



            logging.info(bucket.logits.data)

    def exc(self):

            for epoch in range(self.__variable.EPOCHS):

                logging.info('Epoch: %s', epoch)
                self.__train()
