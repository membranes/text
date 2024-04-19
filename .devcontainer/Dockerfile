# TensorFlow
FROM tensorflow/tensorflow:latest-gpu


# Set application directory <app>
WORKDIR /app


# This command copies requirements.txt into /app
COPY /.devcontainer/requirements.txt /app

RUN pip install --upgrade pip && \
    pip install --rrequirement /app/requirements.txt --no-cache


# Port
EXPOSE 8050


# ENTRYPOINT
ENTRYPOINT ["bash"]