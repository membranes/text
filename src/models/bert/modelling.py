
import logging

import sklearn.metrics as sm
import torch
import torch.utils.data as tu
import transformers
import transformers.modeling_outputs as tm

import src.elements.variable
import src.models.bert.parameters


class Modelling:

    def __init__(self, variable: src.elements.variable.Variable,
                 enumerator: dict, dataloader: tu.DataLoader):
        """

        :param variable:
        :param enumerator: The labels enumerator
        :param dataloader:
        """

        self.__variable = variable
        self.__dataloader = dataloader

        # Parameters
        self.__parameters = src.models.bert.parameters.Parameters()

        # Model
        self.__model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        self.__model.to(self.__parameters.device)

        # Optimisation
        self.__optim = torch.optim.Adam(params=self.__model.parameters(), lr=self.__variable.LEARNING_RATE)

    def __train(self):
        """

        :return:
        """

        # For measures & metrics
        loss_: float = 0
        steps_: int = 0
        accuracy_ = 0

        # For estimates
        __labels: list[torch.Tensor] = []
        __predictions: list[torch.Tensor] = []

        # Preparing a training epoch ...
        self.__model.train()
        logging.info(self.__model.__dict__)

        for epoch in range(self.__variable.EPOCHS):

            logging.info('Epoch: %s', epoch)

            index: int
            batch: dict
            for index, batch in enumerate(self.__dataloader):

                steps_ += 1

                # Parts of the dataset
                inputs_: torch.Tensor = batch['input_ids'].to(self.__parameters.device, dtype = torch.long)
                labels_: torch.Tensor = batch['labels'].to(self.__parameters.device, dtype = torch.long)
                attention_mask_: torch.Tensor = batch['attention_mask'].to(self.__parameters.device, dtype = torch.long)

                # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput
                bucket: tm.TokenClassifierOutput = self.__model(input_ids=inputs_, attention_mask=attention_mask_, labels=labels_)

                # Loss, Tracking Loss Aggregates
                loss_ += bucket.loss.item()
                if (index % 100) == 0:
                    print(f'Average epoch loss, after step {steps_}: {loss_/steps_}')

                # Targets, active targets.
                targets = labels_.view(-1)
                active = labels_.view(-1).ne(100)
                __labels.extend(torch.masked_select(targets, active))

                # Predictions
                logits = bucket.logits.view(-1, self.__model.config.num_labels)
                __predictions.extend(torch.argmax(logits, dim=1))

                # Accuracy
                # Replace this metric
                score: float = sm.accuracy_score(__labels[-1].cpu().numpy(), __predictions[-1].cpu().numpy())
                accuracy_ += score

                # Gradient: Ambiguous
                torch.nn.utils.clip_grad_norm(parameters=self.__model.parameters(),
                                              max_norm=self.__variable.MAX_GRADIENT_NORM)
                self.__optim.zero_grad()
                bucket.loss.backward()
                self.__optim.step()

        logging.info(loss_ / steps_)

    def exc(self):

        self.__train()
