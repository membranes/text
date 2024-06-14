
import transformers


class Parameters:
    """
    
    """

    def __init__(self):

        # Name
        self.pretrained_model_name: str = 'google-bert/bert-base-uncased'

        # Bases
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name)
