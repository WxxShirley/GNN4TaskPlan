# Can Graph Learning Improve Task Planning?

This is the official implementation for paper "Can Graph Learning Improve Task Planning?".

## Table of Contents

- [Can Graph Learning Improve Task Planning?](#can-graph-learning-improve-task-planning?)
   - [Table of Contents](#table-of-contents)
   - [Environment Setup](#environment-setup)
       - [Deploy Open-sourced LLMs](#deploy-open-sourced-llms)
   - [Training-free Methods](#training-free-methods)
       - [Repo Intro](#repo-intro)
       - [Reproducibility](#reproducibility)
   - [Fine-tuning LLMs](#fine-tuning-llms)
   - [TODO](#todo)


## Environment Setup

```shell
pip install -r requirements.txt
```

Run the above command to install required Python packages.

### Deploy Open-sourced LLMs

For running LLM's direct inference or GraphSearch, our codes are implemented as deploying LLMs as API services using [`FastChat`](https://github.com/lm-sys/FastChat) to the `localhost:8008` endpoint. 

* **Installing FastChat** 
  ```shell
  pip3 install "fschat[model_worker,webui]"
  pip3 install vllm
  ```

* **Deploying LLM**
  ```shell
  python3 -m fastchat.serve.controller --host 127.0.0.1 

  # Specify the LLM to be deployed, take CodeLlama-13B as an example
  python3 -m fastchat.serve.vllm_worker --model-path codellama/CodeLlama-13b-Instruct-hf --host 127.0.0.1
  # Commands for other experimental LLMs
  # python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-13b-v1.5 --host 127.0.0.1
  # python3 -m fastchat.serve.vllm_worker --model-path codellama/CodeLlama-7b-Instruct-hf --host 127.0.0.1
  # python3 -m fastchat.serve.vllm_worker --model-path mistralai/Mistral-7B-Instruct-v0.2 --host 127.0.0.1
  # python3 -m fastchat.serve.vllm_worker --model-path baichuan-inc/Baichuan2-13B-Chat --host 127.0.0.1
  
  python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 8008
  ```


## Training-free Methods 

### Repo Intro
Codes of training-free modes are under the **`trainfree`** folder:
* **Direct** LLM's direct inference (default setting - 1-shot in-context learning). Implementation refers to [TaskBench](https://github.com/microsoft/JARVIS/blob/main/taskbench/README.md).
* **GraphSearch** LLM performs iterative search on Task Graph to select an optimal invocation path, including GreedySearch, AdaptiveSearch, and BeamSearch three variants. Implementation refers to [ControlLLM](https://github.com/OpenGVLab/ControlLLM/blob/main/cllm/services/tog/tool.py).
* **SGC** Our training-free SGC method.

Besides, we also provide two improved prompt templates, **2-shot** and **PlaG**, to investigate the orthogonal effectiveness of our method. 

```
├── trainfree
│   ├── direct.py               --> LLM's direct inference
│   ├── direct_diffprompt.py    --> LLM's direct inference under improved prompts, including 1) more in-context learning examples and 2) plan like a graph (PlaG
│   ├── graphsearch.py          --> GraphSearch method
│   └── sgc.py                  --> SGC method
```

### Reproducibility

Running scripts can be found in `trainfree_script.sh`. 

**Hint** You have to first run the Direct Inference to obtain any LLM's direct inference results to faciliate SGC or GraphSearch.



## Fine-tuning LLMs

### Repo Intro 

Codes of fine-tuning LLMs are under the **`finetunellm`** folder:
* **LLM Fine-tune**  Using LoRA to fine-tune a LLM with training data coming from the ground-truth `<user_request, decomposed task steps, task invocation path>` triplets. During experiments, we fine-tune LLMs of different parameter scales, including **[CodeLLaMA-7B](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)** and **[Vicuna-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5)**.
* **Inference based on Fine-tuned LLMs** Making direct inference based on fine-tuned LLMs. You have to specify the LLM's name and the ckpt_dir. 

```
├── finetunellm
│   ├── inference.py         --> Direct inference of fine-tuned LLMs
│   ├── main.py              --> Fine-tuning LLM
│   └── user_prompt.py       --> Instruction Template
```

### Reproducibility

Running scripts can be found in `finetunellm_script.sh`. We use 2 NVIDIA A100-80G GPUs for fine-tuning LLMs.




## TODO 

- [x] [Code] Release all related codes of open-sourced LLMs
- [ ] [Code] Release training-free codes of GPT-series
- [ ] [Docs] Provide a Chinese version README
- [ ] [Code] Provide direct inference results of several LLMs
- [ ] [Resource] Provide ckpt of both GNN and LM+GNN for reproducibility

      
