FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
# LABEL maintainer "Vinker Yael <yael.vinker@mail.huji.ac.il>"

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    cmake \
    vim \
    htop \
    locales \
    unzip \
    wget \
    curl \
    ca-certificates \
    sudo \
    git \
    nano \
    screen \
    gcc \
    python3-setuptools \
    python3-pip \
    python3-dev \
    htop \
    bzip2 \
    libx11-6 \
    libssl-dev \
    libffi-dev \
    parallel \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/hdr_venv
# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n hdr_venv python=3.7

RUN conda install -y -c pytorch \
    cudatoolkit=10.1 \
    "pytorch=1.4.0" \
    "torchvision=0.5.0" \
  && conda clean -ya

RUN conda install -y \
    ipython==6.5.0 \
    matplotlib==3.0.3 \
    plac==0.9.6 \
    py==1.6.0 \
    scipy==1.3.1 \
    tqdm==4.36.1 \
    pathlib==1.0.1 \
    seaborn==0.10.0 \
    scikit-learn==0.22.1 \
    scikit-image==0.16.2 \
    && conda clean -ya

RUN conda install -c conda-forge imageio && conda clean -ya

RUN conda install -c conda-forge jupyterlab && conda clean -ya

# Activate environment in .bashrc.
RUN echo "conda activate hdr_venv" >> /root/.bashrc

RUN pip install \
  imageio \
  torchsummary

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

# Clone unpaired_hdr_tmo.
RUN git clone https://github.com/yael-vinker/unpaired_hdr_tmo.git

WORKDIR /src/unpaired_hdr_tmo

# Run
CMD ["bash", "-c", "python -m torchbeast.polybeast --xpid example"]
