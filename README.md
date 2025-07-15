<br>

This repository hosts the hyperparameter search[^method][^beware] & fine-tuning steps for pre-trained large language model architectures.  It objective is the development of a token classification model in relation to a set of classifications of interest, e.g., geo-political entities, organisations, etc.  

The [numerics package](https://github.com/membranes/numerics) determines the best model.  You may explore, interact with, the model via the [simple interface available here](https://d22j2jhm9iagpk.cloudfront.net/src/c-dispatches-app.html).

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>

[^method]: Herein, hyperparameter search is via the [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer) class of huggingface.co.
[^beware]: [huggingface.co environment variables](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables)