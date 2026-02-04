FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    vim \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY environment.yml /workspace/environment.yml

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "sensory_mapping", "/bin/bash", "-c"]

RUN pip install jupyterlab ipykernel \
    && python -m ipykernel install --name sensory_mapping --display-name "Python (sensory_mapping)"

CMD ["/bin/bash"]
