<br>

## Development Environment

### Remote Development

For this Python project/template, the remote development environment requires

* [Dockerfile](../.devcontainer/Dockerfile)
* [requirements.txt](../.devcontainer/requirements.txt)

An image is built via the command

```shell
docker build . --file .devcontainer/Dockerfile -t text
```

On success, the output of

```shell
docker images
```

should include

<br>

| repository | tag    | image id | created  | size     |
|:-----------|:-------|:---------|:---------|:---------|
| text       | latest | $\ldots$ | $\ldots$ | $\ldots$ |


<br>

Subsequently, run a container, i.e., an instance, of the image `text` via:

<br>

```shell
docker run --rm --gpus all --shm-size=16gb -i -t 
  -p 127.0.0.1:6007:6007 -p 127.0.0.1:6006:6006 
    -p 172.17.0.2:8265:8265 -p 172.17.0.2:6379:6379 -w /app 
	    --mount type=bind,src="$(pwd)",target=/app text
```

or

```shell
docker run --rm --gpus all --shm-size=16gb -i -t 
  -p 6007:6007 -p 6006:6006 -p 8265:8265 -p 6379:6379  
    -w /app --mount type=bind,src="$(pwd)",target=/app text
```

<br>

Herein, `-p 6007:6007` maps the host port `6007` to container port `6007`.  Note, the container's working environment, i.e., -w, must be inline with this project's top directory.  Additionally

* --rm: [automatically remove container](https://docs.docker.com/engine/reference/commandline/run/#:~:text=a%20container%20exits-,%2D%2Drm,-Automatically%20remove%20the)
* -i: [interact](https://docs.docker.com/engine/reference/commandline/run/#:~:text=and%20reaps%20processes-,%2D%2Dinteractive,-%2C%20%2Di)
* -t: [tag](https://docs.docker.com/get-started/02_our_app/#:~:text=Finally%2C%20the-,%2Dt,-flag%20tags%20your)
* -p: [publish](https://docs.docker.com/engine/reference/commandline/run/#:~:text=%2D%2Dpublish%20%2C-,%2Dp,-Publish%20a%20container%E2%80%99s)

<br>

Get the name of the running instance of ``text`` via:

```shell
docker ps --all
```

Never deploy a root container, study the production [Dockerfile](../Dockerfile); cf. [/.devcontainer/Dockerfile](../.devcontainer/Dockerfile)

<br>

### Remote Development & Integrated Development Environments

An IDE (integrated development environment) is a helpful remote development tool.  The **IntelliJ IDEA** set up involves connecting to a machine's Docker [daemon](https://www.jetbrains.com/help/idea/docker.html#connect_to_docker), the steps are

<br>

> * **Settings** $\rightarrow$ **Build, Execution, Deployment** $\rightarrow$ **Docker** $\rightarrow$ **WSL:** {select the linux operating system}
> * **View** $\rightarrow$ **Tool Window** $\rightarrow$ **Services** <br>Within the **Containers** section connect to the running instance of interest, or ascertain connection to the running instance of interest.

<br>

**Visual Studio Code** has its container attachment instructions; study [Attach Container](https://code.visualstudio.com/docs/devcontainers/attach-container).

<br>
<br>


## Code Analysis

The GitHub Actions script [main.yml](../.github/workflows/main.yml) conducts code analysis within a Cloud GitHub Workspace.  Depending on the script, code analysis may occur `on push` to any repository branch, or `on push` to a specific branch.

The sections herein outline remote code analysis.

<br>

### pylint

The directive

```shell
pylint --generate-rcfile > .pylintrc
```

generates the dotfile `.pylintrc` of the static code analyser [pylint](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html).  Analyse a directory via the command

```shell
python -m pylint --rcfile .pylintrc {directory}
```

The `.pylintrc` file of this template project has been **amended to adhere to team norms**, including

* Maximum number of characters on a single line.
  > max-line-length=127

* Maximum number of lines in a module.
  > max-module-lines=135


<br>


### pytest & pytest coverage

The directive patterns

```shell
python -m pytest tests/{directory.name}/...py
pytest --cov-report term-missing  --cov src/{directory.name}/...py tests/{directory.name}/...py
```

for test and test coverage, respectively.


<br>


### flake8

For code & complexity analysis.  A directive of the form

```bash
python -m flake8 --count --select=E9,F63,F7,F82 --show-source --statistics src/...
```

inspects issues in relation to logic (F7), syntax (Python E9, Flake F7), mathematical formulae symbols (F63), undefined variable names (F82).  Additionally

```shell
python -m flake8 --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics src/...
```

inspects complexity.


<br>
<br>



## Modelling

### Data

* [WNUT 2017 (WNUT 2017 Emerging and Rare entity recognition)](https://paperswithcode.com/dataset/wnut-2017-emerging-and-rare-entity)
* [Token Classification & WNUT 2017](https://huggingface.co/docs/transformers/tasks/token_classification)
* [get](https://huggingface.co/datasets/leondz/wnut_17)
* [Few-NERD](https://paperswithcode.com/dataset/few-nerd)
* [get Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd?library=datasets)

<br>
<br>

### Modelling Notes

Word level annotation scheme: <abbr title="Inside, Outside, Beginning">IOB</abbr> Tagging
* [tagtog](https://docs.tagtog.com)
* [doccano](https://github.com/doccano/doccano)

<br>

STEPS
* The Data
* Format vis-à-vis annotation scheme.
* Investigate tag categories imbalances, i.e., cf. the categories tags frequencies vis-à-vis <abbr title="inside">I</abbr>, <abbr title="outside">O</abbr>, & <abbr title="beginning">B</abbr>.
* Beware of token encoding approaches.

<br>

GUIDES
* [transformers](https://huggingface.co/docs/transformers/index)
* [transformers.PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__)
* [serving](https://medium.com/@anthonyproctor/how-to-use-ollama-an-introduction-to-efficient-ai-model-serving-43870d5ae62c)
* [pytorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
* [pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
* [TokenClassifierOutput](https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput)
* [DatasetInfo, Dataset, etc](https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#main-classes)
* [Tensors](https://pytorch.org/docs/stable/tensors.html)
* For tensors that have a single value: [torch.Tensor.item()](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html#torch.Tensor.item)
* Optimisation: [torch.optim](https://pytorch.org/docs/stable/optim.html#module-torch.optim)

<br>

EXTRACTORS
* [pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html), [pip](https://pypi.org/project/pypdf/)
  * [example](https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/)
* [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/), [pip](https://pypi.org/project/PyMuPDF/)
  * [example](https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/)

<br>

File Formats (Note $\rightarrow$ GPT: Generative Pre-trained Transformer)
* GGUF: GPT-Generated Unified Format
* GGML: GPT-Generated Model Language
* [What is GGUF and GGML?](https://medium.com/@phillipgimmi/what-is-gguf-and-ggml-e364834d241c)
* [About GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
* [to GGUF](https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987)
* [to GGUF discussion](https://github.com/ggerganov/llama.cpp/discussions/2948)
* [Hugging Face & GGUF](https://huggingface.co/docs/hub/gguf)

<br>
<br>

### And

* [Freedom of information (FOI) Improvement Plan 2024](https://www.gov.scot/publications/freedom-of-information-foi-improvement-plan-2024/)
* [ELECTRA](https://research.google/blog/more-efficient-nlp-model-pre-training-with-electra/)
* [Large Language Models: A Survey](https://arxiv.org/pdf/2402.06196v1)
* [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training)
* [A survey on recent advances in Named Entity Recognition](https://arxiv.org/html/2401.10825v1)
* [END-TO-END NAMED ENTITY RECOGNITION AND RELATION EXTRACTION USING PRE-TRAINED LANGUAGE MODELS](https://arxiv.org/pdf/1912.13415)

<br>
<br>

## Mathematics, Pre-trained Models, etc

### Mathematics

* [Transformers: Attention is all you need](https://arxiv.org/abs/1706.03762): Re-visit this paper for a basic reminder of the underlying mathematics of [BERT (Bidirectional Encoder Representations from Transformers)](https://arxiv.org/abs/1810.04805)
  * [Annotated @ 2022](https://nlp.seas.harvard.edu/annotated-transformer/)
  * [Annotated @ 2018](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [2.2 Transformer Model $\odot$ Position Information in Transformers: An Overview](https://direct.mit.edu/coli/article/48/3/733/111478/Position-Information-in-Transformers-An-Overview): Study this paper for an understanding of an *unknown* transformer model function.
* [Adam $\odot$ Dive into Deep Learning](https://d2l.ai/chapter_optimization/adam.html)
* [Adam $\odot$ PyTorch](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)
* [ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/abs/1412.6980)
* [KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss)
* [Graph Transformer Networks](https://arxiv.org/abs/1911.06455)
* [Knowledge graph extension with a pre-trained language model via unified learning method](https://dl.acm.org/doi/10.1016/j.knosys.2022.110245)

<br>

### Models

[XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
[DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)

<br>

### Additionally

* [Wordpieces: Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144)
* [Minimisation Algorithms](https://era.ed.ac.uk/handle/1842/4109)
* [A simple illustration of likehood functions in relation to based transformer models](https://etc.cuit.columbia.edu/news/basics-language-modeling-transformers-gpt)
* [Displaying Mathematics Formulae](https://en.wikipedia.org/wiki/Help:Displaying_a_formula)
* [A catalogue of transformer models](https://arxiv.org/html/2302.07730v4)
* [Pattern Recognition & Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
* [Deep Learning](https://www.bishopbook.com)


<br>

### Applications

Named Entity Recognition

* [Annotation Tools](https://www.labellerr.com/blog/7-best-text-annotation-labeling-tools-in-2024/)
* [Doccano](https://microsoft.github.io/nlp-recipes/examples/annotation/Doccano.html)
	* [More](https://doccano.github.io/doccano/)
* [NER (Named Entity Recognition) Annotator](https://github.com/tecoholic/ner-annotator)
* [Acharya for NER (Named Entity Recognition)](https://acharya.astutic.com/docs/intro)
* [gradio & NER (Named Entity Recognition)](https://www.gradio.app/guides/named-entity-recognition)


<br>

### Data Sets

[A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426)


<br>

### Surveys

[AMMUS : A Survey of Transformer-based Pretrained Models in Natural Language Processing](https://arxiv.org/abs/2108.05542)
[A review of pre-trained language models: from BERT, RoBERTa, to ELECTRA, DeBERTa, BigBird, and more](https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/)
[Graph Transformers: A Survey](https://arxiv.org/abs/2407.09777)

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
