<br>

Language

<br>


### Notes

```bash
docker run --rm -i -t -p 127.0.0.1:10000:8888 -w /app --mount type=bind,src="$(pwd)",target=/app 
```

For a standalone exploration of `tensorflow/tensorflow:latest-gpu`
```bash
docker run --rm -i -t -p 127.0.0.1:10000:8888 tensorflow/tensorflow:latest-gpu
```
