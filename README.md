<br>

Language

<br>

### Development Notes

Build

```bash
docker build . --file .devcontainer/Dockerfile --tag entities
```

<br>

For a built image ...

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
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
```

<br>
<br>

### Data

* [WNUT 2017 (WNUT 2017 Emerging and Rare entity recognition)](https://paperswithcode.com/dataset/wnut-2017-emerging-and-rare-entity)
* [Token Classification & WNUT 2017](https://huggingface.co/docs/transformers/tasks/token_classification)
* [get](https://huggingface.co/datasets/leondz/wnut_17)

* [Few-NERD](https://paperswithcode.com/dataset/few-nerd)
* [get Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd?library=datasets)

<br>
<br>

### Modelling Notes

* Word level annotation scheme: <abbr title="Inside, Outside, Beginning">IOB</abbr> Tagging
  * [tagtog](https://docs.tagtog.com)
  * [doccano](https://github.com/doccano/doccano)

* STEPS
  * The Data
  * Format vis-à-vis annotation scheme.
  * Investigate tag categories imbalances, i.e., cf. the categories tags frequencies vis-à-vis <abbr title="inside">I</abbr>, <abbr title="outside">O</abbr>, & <abbr title="beginning">B</abbr>.
  * Beware of token encoding approaches.

* GUIDES
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

* EXTRACTORS
  * [pypdf](https://pypdf.readthedocs.io/en/stable/user/extract-text.html), [pip](https://pypi.org/project/pypdf/)
    * [example](https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/)
  * [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/), [pip](https://pypi.org/project/PyMuPDF/)
    * [example](https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/)


* File Formats (Note $\rightarrow$ GPT: Generative Pre-trained Transformer)
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

<br>
<br>

<br>
<br>

<br>
<br>
