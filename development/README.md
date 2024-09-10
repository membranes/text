<br>

## Development Notes

Build

```bash
docker build . --file .devcontainer/Dockerfile --tag entities
```

<br>

An instance of the image

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 -w /app \
    --mount type=bind,src="$(pwd)",target=/app entities
```

<br>

For a standalone exploration of `pytorch/pytorch:...`

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 
	pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

or

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 
	pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
```

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
