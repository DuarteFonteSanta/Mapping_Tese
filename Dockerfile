FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .

RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    rm -rf /var/lib/apt/lists/*


ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set channel_priority strict && \
    conda config --remove channels defaults || true && \
    conda config --add channels conda-forge && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f environment.yml && \
    conda clean -afy

ENV PATH="/opt/conda/envs/somatosensory_mapping/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=somatosensory_mapping
COPY . .

RUN pip install -e .
EXPOSE 8888

RUN echo '#!/bin/bash\n\
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""' > /app/start-jupyter.sh && \
chmod +x /app/start-jupyter.sh

CMD ["/bin/bash"]
