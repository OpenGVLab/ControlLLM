# ControlLLM [[Paper](https://arxiv.org/abs/2310.17796)]

<!-- ## Description -->

We present ControlLLM, a novel framework that enables large language models (LLMs) to utilize multi-modal tools for solving complex real-world tasks.Despite the remarkable performance of LLMs, they still struggle with tool invocation due to ambiguous user prompts, inaccurate tool selection and parameterization, and inefficient tool scheduling. To overcome these challenges, our framework comprises three key components: (1) a *task decomposer*  that breaks down a complex task into clear subtasks with well-defined inputs and outputs; (2) a *Thoughts-on-Graph (ToG)* paradigm that searches the optimal solution path on a pre-built tool-resource graph, which specifies the parameter and dependency relations among different tools; and (3) an *execution engine with a rich toolbox* that interprets the solution path and runs the tools efficiently on different computational devices. We evaluate our framework on diverse tasks involving image, audio, and video processing, and demonstrate its superior accuracy, efficiency, and versatility compared to existing methods.

## 🤖 Video Demo



## 🗓️ Schedule

- [ ] Release code
  

## Motivation

![Comparison of different paradigms for task planning](https://github.com/OpenGVLab/ControlLLM/assets/13723743/56534638-f8c1-4707-ab16-917df40dfb39)

## 🏠 Overview

![arch](https://github.com/OpenGVLab/ControlLLM/assets/13723743/e5672074-59f7-4260-8ad7-9f373e8e767c)


## 🎁 Features

![features](https://github.com/OpenGVLab/ControlLLM/assets/13723743/9cb12c03-8fc6-4d38-80e1-dedc9568ff14)


## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🖊️ Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@article{2023controlllm,
  title={ControlLLM: Augment Language Models with Tools by Searching on Graphs},
  author={Liu, Zhaoyang and Lai, Zeqiang and Gao Zhangwei and Cui, Erfei and Li, Zhiheng and Zhu, Xizhou and Lu, Lewei and Chen, Qifeng and Qiao, Yu and Dai, Jifeng and Wang Wenhai},
  journal={arXiv preprint arXiv:2305.10601},
  year={2023}
}
```
