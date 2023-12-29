FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive HOME=/root 

RUN apt-get clean && apt-get update && apt install -y python3.10-dev && apt install -y  \
            git libass-dev cmake  libsndfile1-dev tesseract-ocr espeak-ng python3-pip ffmpeg  \
          ninja-build ca-certificates python3.10-tk

# RUN python3 -m pip install --no-cache-dir --upgrade pip && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN ln -sv /usr/bin/python3 /usr/bin/python && python3 -m pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
WORKDIR /root

RUN git clone https://github.com/OpenGVLab/ControlLLM.git

WORKDIR /root/ControlLLM

RUN pip  install --no-cache-dir git+https://github.com/haotian-liu/LLaVA.git 
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip  install --no-cache-dir -r requirements.txt

RUN pip install -e .

EXPOSE 10004
EXPOSE 10005
EXPOSE 10024

