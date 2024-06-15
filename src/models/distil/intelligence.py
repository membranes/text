import transformers
import torch.utils.data as tu

import src.models.distil.parameters
import src.elements.variable
import src.models.distil.parameters

class Intelligence:

    def __init__(self, variable: src.elements.variable.Variable,
                 enumerator: dict, dataloader: tu.DataLoader):

        self.__parameters = src.models.distil.parameters.Parameters()

        transformers.AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=self.__parameters.model_checkpoint,
            **{'num_labels': len(enumerator)})

