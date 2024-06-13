
import logging
import transformers

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

        training_loss = 0
        training_accuracy = 0

        self.__model.train()
        logging.info(self.__model.__dict__)

        '''
        index: int
        batch: dict
        for index, batch in enumerate(self.__dataloader):

            logging.info(batch.keys())
            inputs_ = batch['input_ids'].to(self.__device, dtype = torch.long)
            labels_ = batch['labels'].to(self.__device, dtype = torch.long)
            attention_mask_ = batch['attention_mask'].to(self.__device, dtype = torch.long)
        '''

    def exc(self):

            for epoch in range(self.__variable.EPOCHS):

                logging.info('Epoch: %s', epoch)
                self.__train()
