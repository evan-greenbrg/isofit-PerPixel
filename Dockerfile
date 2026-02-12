# Use lightweight Python image
FROM --platform=$BUILDPLATFORM python:3.11-slim

USER root
RUN apt-get update &&\
    apt-get install --no-install-recommends -y \
      gfortran \
      make \
      nano \
      vim-tiny \
      git &&\
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN git clone https://github.com/evan-greenbrg/isofit.git

WORKDIR /root/isofit
RUN git checkout utils/make_config
RUN git pull origin utils/make_config

WORKDIR /root

RUN pip install -e "isofit[docker]" jupyterlab &&\
    python -m ipykernel install --user --name isofit &&\
    isofit -b . download all &&\
    isofit build

# Make sure ray client is installed
RUN pip install "ray[client]"

# Copy application code
COPY app/ /root/app/

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /root/app

# Ray Dashboard port
EXPOSE 8265

# Expose FastAPI port
EXPOSE 8000

# Start the Jupyterlab server
EXPOSE 8888
