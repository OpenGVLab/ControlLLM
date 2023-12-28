<img src="https://github.com/OpenGVLab/ControlLLM/assets/13723743/5ee0314a-d983-444a-8671-88cc0b52b752" width=10% align="left" /> 

# ControlLLM

*ControlLLM: Augmenting Large Language Models with Tools by Searching on Graphs*

[[Paper](https://arxiv.org/abs/2310.17796)] [[Project Page](https://llava-vl.github.io/)] [[Demo](https://cllm.opengvlab.com)] [[🤗 Space](https://huggingface.co/spaces/OpenGVLab/ControlLLM)]

We present ControlLLM, a novel framework that enables large language models (LLMs) to utilize multi-modal tools for solving complex real-world tasks. Despite the remarkable performance of LLMs, they still struggle with tool invocation due to ambiguous user prompts, inaccurate tool selection and parameterization, and inefficient tool scheduling. To overcome these challenges, our framework comprises three key components: (1) a $\textit{task decomposer}$ that breaks down a complex task into clear subtasks with well-defined inputs and outputs; (2) a $\textit{Thoughts-on-Graph (ToG) paradigm}$ that searches the optimal solution path on a pre-built tool graph, which specifies the parameter and dependency relations among different tools; and (3) an $\textit{execution engine with a rich toolbox}$ that interprets the solution path and runs the tools efficiently on different computational devices. We evaluate our framework on diverse tasks involving image, audio, and video processing, demonstrating its superior accuracy, efficiency, and versatility compared to existing methods.


## 🤖 Video Demo

https://github.com/OpenGVLab/ControlLLM/assets/13723743/cf72861e-0e7b-4c15-89ee-7fa1d838d00f

## 🏠 System Overview

![arch](https://github.com/OpenGVLab/ControlLLM/assets/13723743/dd051971-e5f8-4eaf-96e8-79987ec67ab9#center)

## 🎁 Major Features 
- Image Perception
- Image Editing
- Image Generation
- Video Perception
- Video Editing
- Video Generation
- Audio Perception
- Audio Generation
- Multi-Solution
- Pointing Inputs
- Resource Type Awareness
  
## 🗓️ Schedule

- ✅ (🔥 New) Rlease online [demo](https://cllm.opengvlab.com) and 🤗Hugging Face [space](https://huggingface.co/spaces/OpenGVLab/ControlLLM).
- ✅ (🔥 New) Support [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha), a state-of-the-art method for Text-to-Image synthesis.

## 🛠️Installation

### Basic requirements

* Linux
* Python 3.10+
* PyTorch 2.0+
* CUDA 11.8+

### Clone project

Execute the following command in the root directory:

```bash
git clone https://github.com/OpenGVLab/ControlLLM.git
cd controlllm
```

### Install dependencies

Setup environment:

```bash
conda create -n cllm python=3.10

conda activate cllm

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file):

```bash
pip install git+https://github.com/haotian-liu/LLaVA.git
```

Then install other dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## 👨‍🏫 Get Started

### Step 1: Launch tool services

Please put your personal OpenAI Key and [Weather Key](https://www.visualcrossing.com/weather-api) into the corresponding environment variables. 

😬 Launch all in one endpoint: 
```bash
# openai key
export OPENAI_API_KEY="..."
# openai base
export OPENAI_BASE_URL="..."
# weather api key
export WEATHER_API_KEY="..."
# resource dir
export SERVER_ROOT="./server_resources"

python -m cllm.services.launch --port 10056 --host 0.0.0.0
```

#### Tools as Services

Take image generation as an example, we first launch the service.

```bash

python -m cllm.services.image_generation.launch --port 10011 --host 0.0.0.0

```

Then, we can call the services via python api:

```python
from cllm.services.image_generation.api import *
setup(port=10011)
text2image('A horse')
```


### Step 2: Launch ToG service

```bash
export OPENAI_BASE_URL="..."
export OPENAI_API_KEY="..."
python -m cllm.services.tog.launch --port 10052 --host 0.0.0.0
```

### Step 3: Launch gradio demo

Use `openssl` to generate the certificate:
```shell
mkdir certificate

openssl req -x509 -newkey rsa:4096 -keyout certificate/key.pem -out certificate/cert.pem -sha256 -days 365 -nodes
```

Last, you can launch gradio demo in your server:
```bash
export TOG_PORT=10052
export CLLM_SERVICES_PORT=10056
export CLIENT_ROOT="./client_resources"

export GRADIO_TEMP_DIR="$HOME/.tmp"
export OPENAI_BASE_URL="..."
export OPENAI_API_KEY="..."

python -m cllm.app.gradio --controller "cllm.agents.tog.Controller" --server-port 10003 --https
```

Alternatively, you can set above variables in `run.sh` and launch all services by running:
```bash
bash ./run.sh
```

## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please cite our paper:

```BibTeX
@article{2023controlllm,
  title={ControlLLM: Augment Language Models with Tools by Searching on Graphs},
  author={Liu, Zhaoyang and Lai, Zeqiang and Gao, Zhangwei and Cui, Erfei and Li, Zhiheng and Zhu, Xizhou and Lu, Lewei and Chen, Qifeng and Qiao, Yu and Dai, Jifeng and Wang, Wenhai},
  journal={arXiv preprint arXiv:2305.10601},
  year={2023}
}
```

## 🤝 Acknowledgement
- Thanks to the open source of the following projects:
    [Hugging Face](https://github.com/huggingface) &#8194;
    [LangChain](https://github.com/hwchase17/langchain) &#8194;
    [SAM](https://github.com/facebookresearch/segment-anything) &#8194;
    [Stable Diffusion](https://github.com/CompVis/stable-diffusion) &#8194; 
    [ControlNet](https://github.com/lllyasviel/ControlNet) &#8194; 
    [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) &#8194; 
    [EasyOCR](https://github.com/JaidedAI/EasyOCR)&#8194;
    [ImageBind](https://github.com/facebookresearch/ImageBind) &#8194;
    [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) &#8194;
    [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file) &#8194;
    [Modelscope](https://modelscope.cn/my/overview) &#8194;
    [AudioCraft](https://github.com/facebookresearch/audiocraft) &#8194;
    [Whisper](https://github.com/openai/whisper) &#8194;
    [Llama 2](https://github.com/facebookresearch/llama) &#8194;
    [LLaMA](https://github.com/facebookresearch/llama/tree/llama_v1)&#8194;

--- 
If you want to join our WeChat group, please scan the following QR Code to add our assistant as a Wechat friend:
<p align="center"><img width="300" alt="image" src="https://github.com/OpenGVLab/DragGAN/assets/26198430/e3f0807f-956a-474e-8fd2-1f7c22d73997"></p> 
