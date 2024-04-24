<br>

Language

<br>

### Notes

Build

```bash
docker build . --file .devcontainer/Dockerfile --tag entities
```

<br>

For a built image ...

```bash
docker run --rm -i -t -p 127.0.0.1:10000:8888 -w /app \
    --mount type=bind,src="$(pwd)",target=/app entities
```

<br>

For a standalone exploration of `pytorch/pytorh:...`

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

or

```bash
docker run --rm --gpus all -i -t -p 127.0.0.1:10000:8888 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel
```

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
