
import logging
import tqdm.auto

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
        self.__n_steps = self.__variable.EPOCHS * len(self.__dataloader)
        self.__progress = tqdm.auto.tqdm(iterable=range(self.__n_steps))

        # Parameters
        self.__parameters = src.models.bert.parameters.Parameters()

        # Model
        self.__model = transformers.BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.pretrained_model_name,
            **{'num_labels': len(enumerator)})
        self.__model.to(self.__parameters.device)

        # Optimizing & Scheduling
        self.__optimizer = torch.optim.AdamW(params=self.__model.parameters(), lr=self.__variable.LEARNING_RATE)
        self.__scheduler = transformers.get_scheduler(
            name='linear', optimizer=self.__optimizer, num_warmup_steps=0,
            num_training_steps=self.__n_steps)

    def __train(self):
        """

        :return:
        """

        # Preparing a training epoch ...
        self.__model.train()

        # Train
        for epoch in range(self.__variable.EPOCHS):

            logging.info('Epoch: %s', epoch)

            # For measures & metrics
            step_ = 0
            loss_ = 0
            accuracy_ = 0
            __labels: list[torch.Tensor] = []
            __predictions: list[torch.Tensor] = []

            # By batch
            index: int
            batch: dict
            for index, batch in enumerate(self.__dataloader):

                step_ += 1

                # Parts of the dataset
                inputs_: torch.Tensor = batch['input_ids'].to(self.__parameters.device, dtype = torch.long)
                labels_: torch.Tensor = batch['labels'].to(self.__parameters.device, dtype = torch.long)
                attention_mask_: torch.Tensor = batch['attention_mask'].to(self.__parameters.device, dtype = torch.long)

                # https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput
                bucket: tm.TokenClassifierOutput = self.__model(input_ids=inputs_, attention_mask=attention_mask_, labels=labels_)

                # Loss
                loss = bucket.loss.item()
                loss_ += loss
                loss.backward()

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
                # torch.nn.utils.clip_grad_norm(parameters=self.__model.parameters(),
                #                               max_norm=self.__variable.MAX_GRADIENT_NORM)
                self.__optimizer.step()
                self.__scheduler.step()
                self.__optimizer.zero_grad()
                self.__progress.update(1)

            logging.info(loss_/step_)
            logging.info(accuracy_/step_)


    def exc(self):

        self.__train()
