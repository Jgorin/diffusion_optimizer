FROM python:3.10-slim

ENV DEBIAN_FRONTENT=noninteractive

# RUN set -xe \
#     && apt-get update \
#     && apt-get install -y python3-pip
# RUN pip install --upgrade pip

# add local package to root
WORKDIR /home
COPY diffusion_optimizer diffusion_optimizer

RUN cd /home/diffusion_optimizer; \
    pip install -e . \
    pip install -r requirements.txt

# switch to home
WORKDIR /home
