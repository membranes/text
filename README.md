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
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

or

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
```

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


<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
