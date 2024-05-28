
import transformers


class Parameters:

    def __init__(self):

        # Bases
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path='google-bert/bert-base-uncased')
