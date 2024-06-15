
import torch
import torch.utils.data as tu
import transformers
import transformers.modeling_outputs as tm
import sklearn.metrics as sm

import src.models.bert.parameters


class Validating:

    def __init__(self, model: transformers.PreTrainedModel, dataloader: tu.DataLoader, archetype: dict):
        """

        :param model: The trained model
        :param dataloader: The validation data DataLoader
        :param archetype: The identifiers to label mappings
        """

        # Model
        self.__model = model

        # Data
        self.__dataloader = dataloader

        # Parameters
        self.__parameters = src.models.bert.parameters.Parameters()

    def __validating(self):

        # Preparing for validation stage ...
        self.__model.eval()

        # For measures & metrics
        step_ = 0
        loss_ = 0
        accuracy_ = 0
        __labels: list[torch.Tensor] = []
        __predictions: list[torch.Tensor] = []

        with torch.no_grad():

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
                loss = bucket.loss
                loss_ += loss.item()

                # Targets, active targets.
                targets = labels_.view(-1)
                active = labels_.view(-1).ne(-100)
                original = torch.masked_select(targets, active)
                __labels.extend(original)

                # Predictions, of active targets
                logits = bucket.logits.view(-1, self.__model.config.num_labels)
                maxima = torch.argmax(logits, dim=1)
                predictions = torch.masked_select(maxima, active)
                __predictions.extend(predictions)

                # Accuracy
                # Replace this metric; inappropriate, and probably incorrect arithmetic.
                score: float = sm.accuracy_score(original.cpu().numpy(), predictions.cpu().numpy())
                accuracy_ += score

    def exc(self):

        self.__validating()
